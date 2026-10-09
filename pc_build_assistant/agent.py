"""Agent assembly and the streaming chat handler.

Holds the per-prompt agent cache, the ``<think>`` stripper, and ``chat()``
— the async generator Gradio calls for each user turn. The agent's tools
come from the MCP hardware server (see ``mcp_client``), so the agent is
async throughout. It is rebuilt only when the effective system prompt
changes (i.e. when the injected session context changes), not every turn.
"""

import re
from collections.abc import AsyncGenerator

from langchain.agents import create_agent
from langchain_core.tools import BaseTool

from .config import MAX_HISTORY_MESSAGES, SYSTEM_PROMPT
from .llm import llm
from .mcp_client import toolbox
from .memory import (
    format_context_for_prompt,
    load_session_context,
    save_history,
    save_session_context,
    update_context_from_tool_call,
)

# Friendly status shown while each tool runs. Keys must match the tool
# names the MCP server registers.
TOOL_STATUS: dict[str, str] = {
    "search_hardware_database": "🔍 Searching hardware database…",
    "compare_components":        "⚖️ Comparing components side-by-side…",
    "check_cpu_gpu_bottleneck":  "⚙️ Analysing CPU–GPU bottleneck…",
    "check_current_value":       "💰 Checking current prices…",
    "find_best_value":           "📊 Ranking components by value…",
}
STATUS_FALLBACK = "🔧 Running tool…"
THINKING_STATUS = "💭 Thinking…"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning blocks from model output.

    Handles both completed blocks and, during streaming, an opening
    ``<think>`` whose closing tag hasn't arrived yet — otherwise the
    user would briefly see raw reasoning while the model thinks.

    Args:
        text: Raw streamed text that may contain thinking blocks.

    Returns:
        Cleaned text with all thinking blocks (and any trailing unclosed
        block) removed and whitespace trimmed.
    """
    text = _THINK_RE.sub("", text)        # completed blocks
    # A closing tag with no opening one: the template opened the block in
    # the prompt, so everything before the tag is reasoning.
    text = text.rpartition("</think>")[2]
    text = _THINK_OPEN_RE.sub("", text)   # in-progress block at the end
    return text.strip()


# Agents are cached by their effective system prompt (and tool set, which
# changes only if the MCP server restarts). The only thing that changes
# between turns is the injected session-context block, so rebuilding every
# turn is wasted work — we rebuild only on a real prompt change.
_agent_cache: dict[tuple[str, int], object] = {}


def build_agent(
    tools: list[BaseTool],
    system_prompt: str = SYSTEM_PROMPT,
    model=None,
):
    """Build an agent over the given tools. Also used by the eval script.

    Args:
        tools: The tools to offer, normally from ``toolbox.get_tools()``.
        system_prompt: The system prompt to give the agent.
        model: Chat model to use; defaults to the shared ``llm``.

    Returns:
        A compiled LangChain agent.
    """
    return create_agent(
        model=model or llm,
        tools=tools,
        system_prompt=system_prompt,
    )


def _get_agent(effective_prompt: str, tools: list[BaseTool]):
    """Return a cached agent for the given prompt, building it if needed.

    Args:
        effective_prompt: The full system prompt including any injected
            session context.
        tools: The MCP tools to give the agent.

    Returns:
        A compiled LangChain agent.
    """
    key = (effective_prompt, id(tools))
    cached = _agent_cache.get(key)
    if cached is None:
        cached = build_agent(tools, effective_prompt)
        # Keep the cache small — only the most recent few prompts matter.
        if len(_agent_cache) > 8:
            _agent_cache.clear()
        _agent_cache[key] = cached
    return cached


def _track_tool_calls(ctx: dict, update: dict | None) -> None:
    """Record every tool call in an agent state update into ``ctx``.

    Args:
        ctx: Session context dict (mutated in place).
        update: A ``{"messages": [...]}`` state update or final state;
            None (e.g. a node that changed nothing) is ignored.
    """
    for msg in (update or {}).get("messages", []):
        for call in getattr(msg, "tool_calls", None) or []:
            update_context_from_tool_call(
                ctx, call.get("name", ""), call.get("args") or {},
            )


async def chat(message: str, history: list[dict]) -> AsyncGenerator[str, None]:
    """Stream a response from the agent for a single user turn.

    Yields progressively longer cleaned text as chunks arrive from the
    model.  While the agent is calling tools, a styled status indicator
    is shown to the user.  Tracks tool-call arguments to maintain a
    persistent session context that is injected into the system prompt.
    Falls back to a non-streaming ``ainvoke`` if streaming produces no
    output.  Persists both conversation history and session context to
    disk after the response is complete.

    Args:
        message: The latest user message.
        history: Prior conversation turns as Gradio message dicts.

    Yields:
        Incrementally growing response text (with thinking blocks
        stripped), interspersed with tool-status indicators.
    """
    ctx = load_session_context()

    # ── Build system prompt with session context ────────────────────
    context_block = format_context_for_prompt(ctx)
    effective_prompt = (
        SYSTEM_PROMPT + "\n" + context_block
        if context_block
        else SYSTEM_PROMPT
    )

    # ── Get cached agent (rebuilds only when the prompt changes) ────
    # The first call starts the MCP hardware server.
    session_agent = _get_agent(effective_prompt, await toolbox.get_tools())

    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    # ── Truncate history ────────────────────────────────────────────
    # Keep only the most recent turns, so prompts (and prompt
    # processing) stay short.  The session context injected into the
    # system prompt preserves key facts (budget, discussed components)
    # so older turns can be safely dropped without losing continuity.
    if len(messages) > MAX_HISTORY_MESSAGES:
        messages = messages[-MAX_HISTORY_MESSAGES:]

    partial = ""
    status = ""
    async for mode, data in session_agent.astream(
        {"messages": messages}, stream_mode=["messages", "updates"],
    ):
        # ── Completed model step: record its tool calls ─────────────
        # Streamed chunks carry tool arguments in fragments (the first
        # chunk's args are {} on OpenAI-compatible backends), so the
        # session context is updated from the finished message instead.
        if mode == "updates":
            _track_tool_calls(ctx, (data or {}).get("model"))
            continue

        chunk, metadata = data
        node = metadata.get("langgraph_node", "")

        # ── Tool call requested by the model ────────────────────────
        # Only the first chunk of each call carries its name.
        call_names = [
            c.get("name")
            for c in getattr(chunk, "tool_call_chunks", None) or []
            if c.get("name")
        ]
        if node == "model" and call_names:
            status = TOOL_STATUS.get(call_names[0], STATUS_FALLBACK)
            yield (
                strip_thinking(partial) + "\n\n" + status
                if partial
                else status
            )
            continue

        # ── Model is thinking (reasoning streams separately) ────────
        if (node == "model" and not chunk.content
                and getattr(chunk, "additional_kwargs", {}).get("reasoning_content")):
            if status != THINKING_STATUS:
                status = THINKING_STATUS
                yield (
                    strip_thinking(partial) + "\n\n" + status
                    if partial
                    else status
                )
            continue

        # ── Tool finished executing ─────────────────────────────────
        if node == "tools":
            status = ""
            continue

        # ── Regular model text ──────────────────────────────────────
        if node != "model":
            continue
        if not hasattr(chunk, "content") or not chunk.content:
            continue

        status = ""
        partial += chunk.content
        cleaned = strip_thinking(partial)
        if cleaned:
            yield cleaned

    # Fallback: non-streaming call when streaming produced nothing.
    if not partial:
        response = await session_agent.ainvoke({"messages": messages})
        _track_tool_calls(ctx, response)
        partial = response["messages"][-1].content
        partial = strip_thinking(partial)
        yield partial

    save_history(history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": strip_thinking(partial)},
    ])
    save_session_context(ctx)

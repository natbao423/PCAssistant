"""Agent assembly and the streaming chat handler.

Holds the per-prompt agent cache, the ``<think>`` stripper, and ``chat()``
— the generator Gradio calls for each user turn. The agent is rebuilt only
when the effective system prompt changes (i.e. when the injected session
context changes), not on every turn.
"""

import re
from typing import Generator

from langchain.agents import create_agent

from .config import MAX_HISTORY_MESSAGES, SYSTEM_PROMPT
from .llm import llm
from .memory import (
    format_context_for_prompt,
    load_session_context,
    save_history,
    save_session_context,
    update_context_from_tool_call,
)
from .tools import ALL_TOOLS, STATUS_FALLBACK, TOOL_STATUS

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
    text = _THINK_OPEN_RE.sub("", text)   # in-progress block at the end
    return text.strip()


# Agents are cached by their effective system prompt. The only thing that
# changes between turns is the injected session-context block, so rebuilding
# every turn is wasted work — we rebuild only on a real prompt change.
_agent_cache: dict[str, object] = {}


def _get_agent(effective_prompt: str):
    """Return a cached agent for the given prompt, building it if needed.

    Args:
        effective_prompt: The full system prompt including any injected
            session context.

    Returns:
        A compiled LangChain agent.
    """
    cached = _agent_cache.get(effective_prompt)
    if cached is None:
        cached = create_agent(
            model=llm,
            tools=ALL_TOOLS,
            system_prompt=effective_prompt,
        )
        # Keep the cache small — only the most recent few prompts matter.
        if len(_agent_cache) > 8:
            _agent_cache.clear()
        _agent_cache[effective_prompt] = cached
    return cached


def chat(message: str, history: list[dict]) -> Generator[str, None, None]:
    """Stream a response from the agent for a single user turn.

    Yields progressively longer cleaned text as chunks arrive from the
    model.  While the agent is calling tools, a styled status indicator
    is shown to the user.  Tracks tool-call arguments to maintain a
    persistent session context that is injected into the system prompt.
    Falls back to a synchronous ``invoke`` if streaming produces no
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
    session_agent = _get_agent(effective_prompt)

    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    # ── Truncate history to fit in VRAM ─────────────────────────────
    # Keep only the most recent turns.  Each tool-call exchange can
    # produce very long responses (spec tables, comparisons), so even
    # a few turns can overflow a 16 GB GPU's context budget.  The
    # session context injected into the system prompt preserves key
    # facts (budget, discussed components) so older turns can be
    # safely dropped without losing continuity.
    if len(messages) > MAX_HISTORY_MESSAGES:
        messages = messages[-MAX_HISTORY_MESSAGES:]

    partial = ""
    status = ""
    for chunk, metadata in session_agent.stream(
        {"messages": messages}, stream_mode="messages",
    ):
        node = metadata.get("langgraph_node", "")

        # ── Tool call requested by the model ────────────────────────
        if node == "model" and getattr(chunk, "tool_calls", []):
            call = chunk.tool_calls[0]
            tool_name = call.get("name", "")
            tool_args = call.get("args", {})

            # Update session context from tool arguments.
            update_context_from_tool_call(ctx, tool_name, tool_args)

            status = TOOL_STATUS.get(tool_name, STATUS_FALLBACK)
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

    # Fallback: synchronous invoke when streaming produced nothing.
    if not partial:
        response = session_agent.invoke({"messages": messages})
        partial = response["messages"][-1].content
        partial = strip_thinking(partial)
        yield partial

    save_history(history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": strip_thinking(partial)},
    ])
    save_session_context(ctx)

"""Persistence for chat history and the structured session context.

Two separate concerns live here:

* Chat history: the raw message list shown in the Gradio UI, persisted so
  it survives a page refresh.
* Session context: a small structured dict (budget, use case, discussed
  components, selected parts) extracted from tool-call arguments. It is
  injected into the system prompt so the agent keeps continuity even after
  older turns are truncated from the message history.
"""

import json

from .config import CONTEXT_FILE, HISTORY_FILE


# ── Chat history ────────────────────────────────────────────────────────

def load_history() -> list[dict]:
    """Load the conversation history from disk.

    Returns:
        List of ``{"role": ..., "content": ...}`` message dicts, or an
        empty list if the file is missing or corrupt.
    """
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def save_history(history: list[dict]) -> None:
    """Persist conversation history to disk as pretty-printed JSON.

    Args:
        history: List of ``{"role": ..., "content": ...}`` message dicts.
    """
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


# ── Session context ─────────────────────────────────────────────────────

_DEFAULT_CONTEXT: dict = {
    "budget": None,
    "use_case": None,
    "resolution": None,
    "components_discussed": [],
    "selected_parts": {},
}


def load_session_context() -> dict:
    """Load the persisted session context from disk.

    Returns:
        Context dict with keys for budget, use case, resolution,
        discussed components, and selected parts.  Returns defaults
        if the file is missing or corrupt.
    """
    if CONTEXT_FILE.exists():
        try:
            ctx = json.loads(CONTEXT_FILE.read_text())
            # Ensure all expected keys are present.
            for key, default in _DEFAULT_CONTEXT.items():
                ctx.setdefault(key, default)
            return ctx
        except Exception:
            pass
    return _DEFAULT_CONTEXT.copy()


def save_session_context(ctx: dict) -> None:
    """Persist the session context to disk as JSON.

    Args:
        ctx: Context dict to save.
    """
    CONTEXT_FILE.write_text(json.dumps(ctx, indent=2))


def update_context_from_tool_call(
    ctx: dict,
    tool_name: str,
    tool_args: dict,
) -> dict:
    """Update the session context based on a tool call the model just made.

    Extracts structured info (budget, component names, resolution hints)
    from tool arguments so the context stays current without relying on
    the LLM to self-report.

    Args:
        ctx: Current session context dict (mutated in place).
        tool_name: Name of the tool that was called.
        tool_args: Arguments dict passed to the tool.

    Returns:
        The updated context dict.
    """
    # ── Track discussed components ──────────────────────────────────
    component_keys = [
        "query", "component_name", "component_a", "component_b",
        "cpu_name", "gpu_name",
    ]
    for key in component_keys:
        val = tool_args.get(key)
        if val and val not in ctx["components_discussed"]:
            ctx["components_discussed"].append(val)

    # Keep only the most recent 15 to avoid unbounded growth.
    ctx["components_discussed"] = ctx["components_discussed"][-15:]

    # ── Track budget ────────────────────────────────────────────────
    if "budget" in tool_args:
        try:
            ctx["budget"] = float(tool_args["budget"])
        except (ValueError, TypeError):
            pass

    # ── Track component type as use-case hint ───────────────────────
    comp_type = tool_args.get("component_type", "")
    if comp_type:
        ctx["use_case"] = f"{comp_type.upper()} shopping"

    return ctx


def format_context_for_prompt(ctx: dict) -> str:
    """Render the session context as a compact string for the system prompt.

    Only includes fields that have actual values, keeping the prompt
    overhead minimal.

    Args:
        ctx: Current session context dict.

    Returns:
        A short multi-line string summarising the session, or an empty
        string if nothing has been recorded yet.
    """
    parts: list[str] = []

    if ctx.get("budget"):
        parts.append(f"Budget: ${ctx['budget']:.0f}")
    if ctx.get("use_case"):
        parts.append(f"Use case: {ctx['use_case']}")
    if ctx.get("resolution"):
        parts.append(f"Resolution: {ctx['resolution']}")
    if ctx.get("components_discussed"):
        recent = ctx["components_discussed"][-8:]
        parts.append(f"Discussed: {', '.join(recent)}")
    if ctx.get("selected_parts"):
        picks = [f"{k}: {v}" for k, v in ctx["selected_parts"].items()]
        parts.append(f"Selected: {', '.join(picks)}")

    if not parts:
        return ""
    return "USER CONTEXT (from this session):\n" + "\n".join(parts)

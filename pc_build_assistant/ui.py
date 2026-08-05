"""Gradio UI definition.

``build_ui()`` returns the assembled Blocks app. Launching it — and
passing the CSS — is left to the entry point in ``app.py``. As of
Gradio 6.0, ``css`` must go to ``launch()``, not the ``Blocks``
constructor.

The theme follows the OS light/dark setting. Both palettes are defined
in ``styles.py`` (``:root`` for light, ``.dark`` for dark), so no JS
theme forcing is needed.
"""

import gradio as gr

from .agent import chat
from .config import CONTEXT_FILE, EXAMPLES, HISTORY_FILE
from .memory import load_history
from .styles import CUSTOM_CSS  # re-exported for app.py


def _clear_history() -> list:
    """Delete persisted history and session context, returning empty chat."""
    HISTORY_FILE.unlink(missing_ok=True)
    CONTEXT_FILE.unlink(missing_ok=True)
    return []


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app (not yet launched).

    Returns:
        The configured ``gr.Blocks`` instance.
    """
    with gr.Blocks(title="PC Build Assistant") as demo:
        gr.Markdown(
            "<div style='text-align:center; padding: 8px 0 2px;'>"
            "<span class='pcba-title'>PC Build Assistant</span><br>"
            "<span class='pcba-tagline'>"
            "Specs · comparisons · bottleneck analysis · build advice "
            "— powered by local AI</span></div>"
        )

        chatbot = gr.Chatbot(
            height=500,
            placeholder=(
                "<div style='text-align:center; padding:40px 0;'>"
                "<span class='pcba-ph-main'>What are you building?</span><br>"
                "<span class='pcba-ph-sub'>Ask about GPUs, CPUs, "
                "bottlenecks, or prices</span></div>"
            ),
        )

        gr.ChatInterface(fn=chat, chatbot=chatbot, examples=EXAMPLES)

        clear_btn = gr.Button("Clear History", variant="secondary")
        clear_btn.click(fn=_clear_history, outputs=chatbot)

        demo.load(fn=load_history, outputs=chatbot)

    return demo
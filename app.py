"""Entry point for the PC Build Assistant.

Run with:  python app.py

The UI theme follows your OS light/dark setting. To force one, append
?__theme=light or ?__theme=dark to the URL.

Requires Ollama running (embeddings, and the chat model when
CHAT_BACKEND == "ollama"). For the llama-server backend, see
SETUP_NOTES.md for the startup command and run order.
"""

from pc_build_assistant.ui import CUSTOM_CSS, build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(css=CUSTOM_CSS)
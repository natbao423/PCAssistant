"""Shared model, embeddings, and retriever instances.

These are module-level singletons built once on import. Tools import
``retriever`` from here, and the agent layer imports ``llm``. Centralising
them in one module avoids circular imports between tools and the agent.

Importing this module connects to the embeddings backend and builds (or
loads) the vector store, so import it only when you actually intend to run
the app, not in unit tests of pure helpers.
"""

from langchain_ollama import OllamaEmbeddings

from .config import (
    CHAT_BACKEND,
    CHAT_MODEL,
    EMBED_MODEL,
    LLAMA_SERVER_URL,
    OLLAMA_CHAT_MODEL,
)
from .data import build_vectorstore


def _build_llm():
    """Construct the chat model for the backend named in the config.

    Returns:
        A LangChain chat model instance (ChatOllama or ChatOpenAI).
    """
    if CHAT_BACKEND == "ollama":
        # Simple local path. Uses a model from `ollama list`. Slower and
        # less reliable than the 35B, but needs no llama.cpp setup — good
        # for confirming the app works end to end.
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0.1)

    if CHAT_BACKEND == "llama_server":
        # Qwen3.6-35B-A3B served by llama.cpp's llama-server (see
        # SETUP_NOTES.md). OpenAI-compatible endpoint, so ChatOpenAI.
        # Thinking is left ON — Qwen3.6 was trained to carry reasoning
        # across tool-calling turns; strip_thinking() hides the <think>
        # blocks from the user.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=CHAT_MODEL,
            base_url=LLAMA_SERVER_URL,
            api_key="not-needed",      # llama-server ignores this
            temperature=0.6,           # Unsloth "Precise" preset
            top_p=0.95,
            max_tokens=2048,
            extra_body={"top_k": 20, "min_p": 0.0},
        )

    raise ValueError(
        f"Unknown CHAT_BACKEND {CHAT_BACKEND!r} — "
        f"use 'ollama' or 'llama_server'."
    )


llm = _build_llm()

# Embeddings always run on Ollama — nomic-embed-text is tiny.
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# Vector store & retriever (built or loaded once at import).
vectorstore = build_vectorstore(embeddings)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20},
)
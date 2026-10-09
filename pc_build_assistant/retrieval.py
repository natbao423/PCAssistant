"""Embeddings, vector store, and retriever for the hardware database.

Built lazily on first use rather than on import, so importing the tools
(e.g. to compute expected values in the evals) doesn't need Ollama until
a search actually runs. Only the MCP server process calls these.
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

from .config import EMBED_MODEL
from .data import build_vectorstore


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Build or load the ChromaDB store (once per process).

    Embeddings always run on Ollama — nomic-embed-text is tiny.
    """
    return build_vectorstore(OllamaEmbeddings(model=EMBED_MODEL))


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    """MMR retriever over the hardware store."""
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )

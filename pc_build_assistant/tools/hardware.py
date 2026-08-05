"""Hardware database search tool.

Combines exact model-name matching against the vector store's metadata
with semantic (MMR) retrieval, labelling each so the model knows which
results to trust. The labelled/​"NOT FOUND" output is deliberately blunt
to stop smaller models filling gaps from memory.
"""

import re

from langchain_core.documents import Document

from ..llm import retriever, vectorstore


def _direct_model_lookup(query: str) -> list[Document]:
    """Find vector-store documents by exact model-name substring match.

    Complements the semantic retriever by scanning every stored document's
    ``model`` metadata field for known hardware name patterns (e.g.
    ``RTX 5080``, ``Ryzen 7 9800X3D``).

    Args:
        query: Free-text user query that may contain model names.

    Returns:
        De-duplicated list of matching Documents, ordered by first
        pattern match.
    """
    patterns = re.findall(
        r"(?:RTX|RX|Ryzen \d|Core (?:Ultra )?\w+|i\d)[- ]?\w*(?:\s\w+)?",
        query,
        re.IGNORECASE,
    )
    if not patterns:
        return []

    all_data = vectorstore.get(include=["documents", "metadatas"])
    hits: list[Document] = []
    seen_ids: set[tuple] = set()

    for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
        model_field = meta.get("model", "")
        for pat in patterns:
            if pat.lower().strip() in model_field.lower():
                key = (meta.get("source"), meta.get("row"))
                if key not in seen_ids:
                    seen_ids.add(key)
                    hits.append(Document(page_content=doc_text, metadata=meta))

    return hits


def search_hardware_database(query: str) -> str:
    """Search the GPU and CPU database for specs, features, and compatibility.

    Combines direct model-name matching (high confidence) with semantic
    retrieval (lower confidence) and clearly labels each.  Returns at
    most 6 documents to keep context tight.

    Args:
        query: Natural-language question about hardware.

    Returns:
        Labelled document contents, or a not-found message.  If no
        exact model match is found, the response says so explicitly.
    """
    direct_results = _direct_model_lookup(query)
    semantic_results = retriever.invoke(query)

    seen: set[tuple] = set()
    exact: list[str] = []
    related: list[str] = []

    for doc in direct_results:
        key = (doc.metadata.get("source"), doc.metadata.get("row"))
        if key not in seen:
            seen.add(key)
            exact.append(doc.page_content)

    for doc in semantic_results:
        key = (doc.metadata.get("source"), doc.metadata.get("row"))
        if key not in seen:
            seen.add(key)
            related.append(doc.page_content)

    if not exact and not related:
        return (
            f"NOT FOUND: No component matching '{query}' exists in the "
            f"database.  Do NOT guess specs — tell the user this "
            f"component is not in the database."
        )

    parts: list[str] = []
    if exact:
        parts.append("=== EXACT MATCHES (use these) ===")
        parts.extend(exact[:4])
    if related:
        parts.append("=== RELATED (only use if relevant to the query) ===")
        parts.extend(related[:2])

    if not exact:
        parts.insert(
            0,
            f"WARNING: No exact match for '{query}'.  The results below "
            f"are semantically similar but may not be the component the "
            f"user asked about.  If none match, tell the user the "
            f"component is not in the database.",
        )

    return "\n\n".join(parts)

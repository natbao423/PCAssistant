"""Data access: CSV loaders, document conversion, and the vector store.

This module owns everything that reads the hardware CSVs from disk or
builds the ChromaDB index. It has no dependency on the LLM or tools, so
it can be imported and tested in isolation.
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from .config import (
    BASE_DIR,
    CHROMA_PATH,
    CSV_SOURCES,
    INDEX_RECORD,
    INDEX_VERSION,
)
from .matching import Mention, ModelIndex, same_model


def load_gpu_df() -> pd.DataFrame:
    """Load the unified GPU specs-and-features CSV into a DataFrame.

    Returns:
        DataFrame with one row per GPU model, containing spec and feature
        columns sourced from ``all_gpus.csv``.
    """
    return pd.read_csv(BASE_DIR / "all_gpus.csv")


def load_cpu_df() -> pd.DataFrame:
    """Load the CPU specs CSV into a DataFrame.

    Returns:
        DataFrame with one row per CPU model, sourced from ``cpus.csv``.
    """
    return pd.read_csv(BASE_DIR / "cpus.csv")


def find_row(df: pd.DataFrame, name: str) -> tuple[pd.Series, Mention] | None:
    """Find the spec row a component name refers to.

    Uses token matching (see ``matching``), so "Ryzen 9 9950X" finds the
    9950X rather than the 9950X3D, and "GeForce RTX 5080" finds the RTX 5080.

    Args:
        df: A spec DataFrame with a ``Model`` column.
        name: Component name as written by the user or model.

    Returns:
        ``(row, mention)``, or None if no model in ``df`` matches.
    """
    mention = ModelIndex(df["Model"]).resolve(name)
    if mention is None:
        return None
    return df[df["Model"] == mention.best].iloc[0], mention


def find_price_row(price_df: pd.DataFrame, model: str) -> pd.Series | None:
    """Return the price row for an exact model name, or None.

    Args:
        price_df: Contents of ``current_prices.csv``.
        model: Canonical model name from a spec CSV.

    Returns:
        The matching row, or None if the model has no price entry.
    """
    if price_df.empty:
        return None
    hits = price_df[price_df["Model"].map(lambda m: same_model(m, model))]
    return None if hits.empty else hits.iloc[0]


def known_models() -> ModelIndex:
    """Index of every GPU and CPU model across both spec CSVs."""
    return ModelIndex(
        list(load_gpu_df()["Model"]) + list(load_cpu_df()["Model"])
    )


def load_csv_docs(filepath: Path, category: str) -> list[Document]:
    """Convert a hardware CSV file into LangChain Documents for indexing.

    Each row becomes a single Document whose ``page_content`` lists every
    non-NaN column as ``"column: value"`` lines, prefixed with the category
    tag (e.g. ``[gpu_specs_and_features]``).

    Args:
        filepath: Path to the CSV file.
        category: Label used to tag every document (stored in metadata too).

    Returns:
        List of Documents ready for insertion into the vector store.
    """
    df = pd.read_csv(filepath)
    docs: list[Document] = []
    for i, row in df.iterrows():
        content = f"[{category}]\n" + "\n".join(
            f"{col}: {val}" for col, val in row.items() if pd.notna(val)
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source": filepath.name,
                "row": i,
                "category": category,
                "model": str(row.get("Model", "")),
            },
        ))
    return docs


def build_vectorstore(embeddings: OllamaEmbeddings) -> Chroma:
    """Build or load the ChromaDB vector store from hardware CSVs.

    The store is rebuilt from scratch whenever the set of source CSV files
    changes or ``INDEX_VERSION`` is bumped.  Otherwise the existing persisted
    store is loaded directly.

    Args:
        embeddings: Embedding model used for document vectorisation.

    Returns:
        A ready-to-query Chroma vector store instance.
    """
    expected_files = sorted(f for f, _ in CSV_SOURCES if (BASE_DIR / f).exists())
    needs_rebuild = True

    if Path(CHROMA_PATH).exists() and INDEX_RECORD.exists():
        try:
            record = json.loads(INDEX_RECORD.read_text())
            if (isinstance(record, dict)
                    and record.get("files") == expected_files
                    and record.get("version") == INDEX_VERSION):
                needs_rebuild = False
        except Exception:
            pass

    if needs_rebuild:
        if Path(CHROMA_PATH).exists():
            shutil.rmtree(CHROMA_PATH)

        docs: list[Document] = []
        for filename, category in CSV_SOURCES:
            path = BASE_DIR / filename
            if path.exists():
                docs.extend(load_csv_docs(path, category))

        store = Chroma.from_documents(
            docs, embeddings, persist_directory=CHROMA_PATH,
        )
        Path(CHROMA_PATH).mkdir(exist_ok=True)
        INDEX_RECORD.write_text(json.dumps({
            "files": expected_files,
            "version": INDEX_VERSION,
        }))
        # stderr, not stdout: the MCP server's stdio transport owns stdout.
        print(
            f"Vector store rebuilt (v{INDEX_VERSION}) with {len(docs)} "
            f"documents from {expected_files}",
            file=sys.stderr,
        )
    else:
        store = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=embeddings,
        )
        print("Loaded existing vector store.", file=sys.stderr)

    return store

"""FAISS vector store — build, save, and load operations."""
import logging
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def build_faiss(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Build a FAISS index from document chunks and an embedding model.

    Raises:
        ValueError: If the chunks list is empty.
        RuntimeError: If embedding or index creation fails.
    """
    if not chunks:
        raise ValueError(
            "Cannot build a FAISS index from an empty chunk list. "
            "Check that the PDF was chunked successfully."
        )
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build FAISS index: {exc}"
        ) from exc

    logger.info("Built FAISS index with %d vectors", len(chunks))
    return vectorstore


def save_faiss(vectorstore: FAISS, index_dir: str) -> None:
    """
    Persist a FAISS vector store to disk.

    Raises:
        RuntimeError: If the directory cannot be created or the save fails.
    """
    try:
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(index_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save FAISS index to '{index_dir}': {exc}"
        ) from exc

    logger.info("Saved FAISS index to '%s'", index_dir)


def load_faiss(index_dir: str, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Load a previously saved FAISS vector store from disk.

    Raises:
        FileNotFoundError: If the index directory or files are missing.
        RuntimeError: If the index files are corrupted or incompatible.
    """
    index_path = Path(index_dir)
    if not index_path.exists() or not (index_path / "index.faiss").exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{index_dir}'. "
            "Please upload a PDF and click 'Build Index' first."
        )

    try:
        vectorstore = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load FAISS index from '{index_dir}'. "
            "The index may be corrupted. Try rebuilding it.\n"
            f"Original error: {exc}"
        ) from exc

    logger.info("Loaded FAISS index from '%s'", index_dir)
    return vectorstore

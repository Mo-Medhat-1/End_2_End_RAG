"""FAISS vector store operations."""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def build_faiss(chunks: List[Document], embeddings) -> FAISS:
    """Build a FAISS index from document chunks and an embedding model."""
    if not chunks:
        raise ValueError("No chunks provided to build FAISS index.")
    return FAISS.from_documents(chunks, embeddings)


def save_faiss(vectorstore: FAISS, index_dir: str) -> None:
    """Persist a FAISS vector store to disk."""
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_dir)


def load_faiss(index_dir: str, embeddings) -> FAISS:
    """Load a previously saved FAISS vector store from disk."""
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def build_faiss(chunks: List[Document], embeddings):
    if not chunks:
        raise ValueError("No chunks provided to build FAISS index.")
    return FAISS.from_documents(chunks, embeddings)


def save_faiss(vectorstore, index_dir: str):
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_dir)


def load_faiss(index_dir: str, embeddings):
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )

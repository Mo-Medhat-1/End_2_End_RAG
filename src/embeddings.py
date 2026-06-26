"""Embedding model configuration."""
import os
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """
    Initialize the HuggingFace embedding model.

    Model name is read from the EMBEDDING_MODEL environment variable,
    defaulting to 'sentence-transformers/all-MiniLM-L6-v2'.
    """
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)

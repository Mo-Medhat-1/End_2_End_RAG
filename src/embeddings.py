"""Embedding model configuration and initialization."""
import os
import logging

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and return the HuggingFace sentence-embedding model.

    The model name is read from the EMBEDDING_MODEL environment variable,
    defaulting to 'sentence-transformers/all-MiniLM-L6-v2'.

    Raises:
        RuntimeError: If the model cannot be loaded (wrong name, no internet, etc.)
    """
    model_name = os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL)
    logger.info("Loading embedding model: %s", model_name)

    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}'.\n"
            "Check your internet connection and that EMBEDDING_MODEL is a valid "
            "sentence-transformers model name.\n"
            f"Original error: {exc}"
        ) from exc

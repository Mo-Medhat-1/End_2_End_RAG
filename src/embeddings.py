import os
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)

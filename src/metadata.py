"""Page-level metadata enrichment for document chunks."""
from typing import List
from langchain_core.documents import Document


def describe_page(text: str, max_words: int = 18) -> str:
    """
    Generate a short description for a page based on its text content.

    Uses a deterministic approach: extracts the first N words as a summary.
    Can be replaced with an LLM-based summarizer for richer descriptions.
    """
    clean = " ".join(text.replace("\n", " ").split())
    if not clean:
        return "Empty or OCR-unreadable page"
    words = clean.split()[:max_words]
    return " ".join(words)


def add_page_descriptions(docs: List[Document]) -> List[Document]:
    """
    Enrich each document with a short text description in its metadata.

    Adds a 'page_description' field to every document's metadata dict.
    """
    for doc in docs:
        doc.metadata["page_description"] = describe_page(doc.page_content)
    return docs

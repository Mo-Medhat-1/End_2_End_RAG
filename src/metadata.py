from typing import List
from langchain_core.documents import Document


def describe_page(text: str, max_words: int = 18) -> str:
    """
    Simple deterministic page description.
    In production, this can be replaced with an LLM summarizer.
    """
    clean = " ".join(text.replace("\n", " ").split())
    if not clean:
        return "Empty or OCR-unreadable page"
    words = clean.split()[:max_words]
    return " ".join(words)


def add_page_descriptions(docs: List[Document]) -> List[Document]:
    """
    Adds a short description to every page-level document.
    """
    for doc in docs:
        doc.metadata["page_description"] = describe_page(doc.page_content)
    return docs


# Task 
## 1- write scripts  / repo structre / Setups .
## 2- Apply metadata script using llm summarizer instead of the simple deterministic one.

"""PDF document loading utilities."""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf_pages(file_path: str) -> List[Document]:
    """
    Load a text-based PDF page by page.

    Each page becomes a LangChain Document with enriched metadata
    including source filename, file path, and 1-indexed page number.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    loader = PyPDFLoader(str(path))
    docs = loader.load()

    for doc in docs:
        page = int(doc.metadata.get("page", 0)) + 1
        doc.metadata.update({
            "source": path.name,
            "file_path": str(path),
            "page_number": page,
            "loader": "PyPDFLoader",
        })
    return docs

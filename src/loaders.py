"""PDF document loading utilities."""
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)


def load_pdf_pages(file_path: str) -> List[Document]:
    """
    Load a text-based PDF page by page.

    Each page becomes a LangChain Document with enriched metadata:
    source filename, absolute file path, 1-indexed page number, and loader name.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the PDF is empty (zero pages extracted).
        RuntimeError: If the PDF is corrupted, encrypted, or unreadable.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    try:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read '{path.name}'. The file may be corrupted, "
            f"password-protected, or not a valid PDF.\nOriginal error: {exc}"
        ) from exc

    if not docs:
        raise ValueError(
            f"No pages were extracted from '{path.name}'. "
            "The PDF may be empty or contain only images (try enabling OCR)."
        )

    for doc in docs:
        page = int(doc.metadata.get("page", 0)) + 1
        doc.metadata.update(
            {
                "source": path.name,
                "file_path": str(path),
                "page_number": page,
                "loader": "PyPDFLoader",
            }
        )

    logger.info("Loaded %d pages from '%s'", len(docs), path.name)
    return docs

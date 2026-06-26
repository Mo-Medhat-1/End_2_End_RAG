"""PDF document loading utilities — powered by PyMuPDF."""
import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf_pages(file_path: str) -> List[Document]:
    """
    Load a PDF page-by-page using PyMuPDF.

    PyMuPDF extracts text with significantly higher fidelity than PyPDF,
    preserving layout, handling complex fonts, and supporting a wider range
    of PDF specifications.

    Each page becomes a LangChain Document with metadata:
    source filename, absolute path, 1-indexed page number, loader name.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file is not a .pdf or has zero pages.
        RuntimeError:      If the PDF is corrupted, encrypted, or unreadable.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix!r}")

    try:
        pdf = fitz.open(str(path))
    except fitz.FileDataError as exc:
        raise RuntimeError(
            f"Cannot open '{path.name}'. "
            "The file may be corrupted or not a valid PDF."
        ) from exc
    except fitz.PasswordRequired:
        raise RuntimeError(
            f"'{path.name}' is password-protected. "
            "Remove the password and try again."
        )

    if pdf.page_count == 0:
        raise ValueError(
            f"'{path.name}' has zero pages. The PDF may be empty."
        )

    docs: List[Document] = []
    for i, page in enumerate(pdf):
        text = page.get_text("text")  # plain text, preserves reading order
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "file_path": str(path),
                    "page_number": i + 1,
                    "loader": "PyMuPDF",
                },
            )
        )

    pdf.close()

    if not docs:
        raise ValueError(
            f"No pages were extracted from '{path.name}'. "
            "The PDF may contain only images — try enabling OCR."
        )

    logger.info("Loaded %d pages from '%s'", len(docs), path.name)
    return docs

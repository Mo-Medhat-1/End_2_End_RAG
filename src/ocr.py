"""OCR utilities for scanned PDF documents."""
import logging
from pathlib import Path
from typing import List

from PIL import Image
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def ocr_pdf_pages(file_path: str, scale: int = 2) -> List[Document]:
    """
    Extract text from scanned PDF pages using Tesseract OCR.

    Renders each page as an image via pypdfium2, then runs Tesseract
    OCR to extract text content.

    Args:
        file_path: Path to the PDF file.
        scale:     Rendering resolution multiplier (higher = better quality, slower).

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If Tesseract is not installed on the system.
        RuntimeError: If the PDF cannot be opened (corrupted / password-protected).
    """
    import pytesseract  # local import — only required when OCR is actually used

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "pypdfium2 is not installed. Add it to requirements.txt."
        ) from exc

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Validate Tesseract availability early with a clear error message.
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or not found on the system PATH.\n"
            "• Docker / Streamlit Cloud: ensure 'tesseract-ocr' is in packages.txt.\n"
            "• Windows local: download from https://github.com/UB-Mannheim/tesseract/wiki"
        ) from exc

    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot open '{path.name}' for OCR. "
            "The file may be corrupted or password-protected.\n"
            f"Original error: {exc}"
        ) from exc

    docs: List[Document] = []
    for i, page in enumerate(pdf):
        try:
            bitmap = page.render(scale=scale).to_pil()
            raw_text = pytesseract.image_to_string(bitmap)
            text = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else str(raw_text)
        except Exception as exc:
            logger.warning("OCR failed on page %d of '%s': %s", i + 1, path.name, exc)
            text = ""

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "file_path": str(path),
                    "page_number": i + 1,
                    "loader": "OCR",
                },
            )
        )

    logger.info("OCR extracted %d pages from '%s'", len(docs), path.name)
    return docs


def needs_ocr(docs: List[Document], min_chars_per_page: int = 40) -> bool:
    """
    Determine if a PDF needs OCR by checking average text density.

    Returns True if more than half the pages have fewer characters
    than ``min_chars_per_page``.
    """
    if not docs:
        return True
    weak_pages = sum(
        len(d.page_content.strip()) < min_chars_per_page for d in docs
    )
    return weak_pages / len(docs) > 0.5
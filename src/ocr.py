"""OCR utilities for scanned PDF documents."""
from pathlib import Path
from typing import List
from PIL import Image
import pytesseract
import pypdfium2 as pdfium
from langchain_core.documents import Document


def ocr_pdf_pages(file_path: str, scale: int = 2) -> List[Document]:
    """
    Extract text from scanned PDF pages using OCR.

    Renders each page as an image via pypdfium2, then runs
    Tesseract OCR to extract text content.

    Requires: tesseract-ocr installed on the system.
    """
    path = Path(file_path)
    pdf = pdfium.PdfDocument(str(path))
    docs: List[Document] = []

    for i, page in enumerate(pdf):
        bitmap = page.render(scale=scale).to_pil()
        raw_text = pytesseract.image_to_string(bitmap)
        text_str = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else str(raw_text)
        docs.append(Document(
            page_content=text_str,
            metadata={
                "source": path.name,
                "file_path": str(path),
                "page_number": i + 1,
                "loader": "OCR",
            }
        ))

    return docs


def needs_ocr(docs: List[Document], min_chars_per_page: int = 40) -> bool:
    """
    Determine if a PDF needs OCR by checking text density.

    Returns True if more than half the pages have fewer
    characters than the threshold.
    """
    if not docs:
        return True
    weak_pages = sum(len(d.page_content.strip()) < min_chars_per_page for d in docs)
    return weak_pages / len(docs) > 0.5
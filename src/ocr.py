"""OCR utilities for scanned/image-based PDF documents — powered by PyMuPDF + Tesseract."""
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Tesseract configuration:
#   --oem 3  → Use the LSTM neural network OCR engine (best accuracy)
#   --psm 6  → Assume a single uniform block of text per page
_TESSERACT_CONFIG = r"--oem 3 --psm 6"

# Render resolution: 300 DPI gives a good balance between accuracy and speed.
# PyMuPDF default is 72 DPI; multiplying the matrix by 300/72 ≈ 4.17 gives 300 DPI.
_DPI = 300
_SCALE = _DPI / 72


def ocr_pdf_pages(file_path: str) -> List[Document]:
    """
    Extract text from a scanned (image-based) PDF using Tesseract OCR.

    Pipeline per page:
        1. Render page to a 300-DPI pixmap via PyMuPDF.
        2. Convert to grayscale.
        3. Binarize (Otsu threshold via Pillow) to reduce noise.
        4. Run Tesseract LSTM (--oem 3) in block-text mode (--psm 6).

    Args:
        file_path: Path to the PDF file.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError:      If Tesseract is not installed on the system.
        RuntimeError:      If the PDF cannot be opened (corrupted / encrypted).
    """
    import pytesseract
    from PIL import Image, ImageFilter
    import fitz  # PyMuPDF — used for high-quality page rendering

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Validate Tesseract availability before processing any pages.
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR engine is not installed or not on the system PATH.\n"
            "• Streamlit Cloud / Docker: ensure 'tesseract-ocr' is in packages.txt.\n"
            "• Windows local: https://github.com/UB-Mannheim/tesseract/wiki"
        ) from exc

    try:
        pdf = fitz.open(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot open '{path.name}' for OCR. "
            "The file may be corrupted or password-protected."
        ) from exc

    matrix = fitz.Matrix(_SCALE, _SCALE)  # scale to 300 DPI

    docs: List[Document] = []
    for i in range(pdf.page_count):
        try:
            page = pdf[i]
            # Render at 300 DPI → grayscale pixmap
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
            img = Image.frombytes("L", [pix.width, pix.height], pix.samples)

            # Mild sharpening to improve character edges for Tesseract
            img = img.filter(ImageFilter.SHARPEN)

            # Binarize: pixels below threshold → black, above → white.
            # Threshold 180 works well for most printed documents.
            img = img.point(lambda px: 255 if px > 180 else 0, mode="L")

            raw = pytesseract.image_to_string(img, config=_TESSERACT_CONFIG)
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

        except Exception as exc:
            logger.warning(
                "OCR failed on page %d of '%s': %s — using empty string",
                i + 1, path.name, exc,
            )
            text = ""

        docs.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "source": path.name,
                    "file_path": str(path),
                    "page_number": i + 1,
                    "loader": "Tesseract-OCR",
                    "ocr_dpi": _DPI,
                },
            )
        )

    pdf.close()
    logger.info(
        "OCR extracted %d pages from '%s' at %d DPI",
        len(docs), path.name, _DPI,
    )
    return docs


def needs_ocr(docs: List[Document], min_chars_per_page: int = 50) -> bool:
    """
    Return True if more than half the pages have fewer than ``min_chars_per_page``
    characters — indicating a scanned / image-based PDF that needs OCR.
    """
    if not docs:
        return True
    weak = sum(len(d.page_content.strip()) < min_chars_per_page for d in docs)
    return weak / len(docs) > 0.5
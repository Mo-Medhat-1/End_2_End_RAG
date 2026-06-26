"""End-to-end PDF ingestion pipeline."""
import logging
from pathlib import Path

from src.loaders import load_pdf_pages
from src.ocr import ocr_pdf_pages, needs_ocr
from src.metadata import add_page_descriptions
from src.chunking import semantic_like_chunk_docs
from src.embeddings import get_embeddings
from src.vectorstore import build_faiss, save_faiss

logger = logging.getLogger(__name__)


def ingest_pdf_to_faiss(
    pdf_path: str,
    index_dir: str = "vectorstore/faiss_index",
    use_ocr_if_needed: bool = True,
) -> dict:
    """
    Full ingestion pipeline: PDF → pages → metadata → chunks → embeddings → FAISS.

    Steps:
        1. Load PDF pages as LangChain Documents
        2. Run Tesseract OCR if text extraction yields weak results
        3. Enrich each page with a short deterministic description
        4. Chunk documents with overlap for retrieval
        5. Build and persist a FAISS vector store

    Args:
        pdf_path:         Path to the source PDF file.
        index_dir:        Directory where the FAISS index will be saved.
        use_ocr_if_needed: Run OCR when text content is sparse.

    Returns:
        dict with keys: ``pages``, ``chunks``, ``index_dir``.

    Raises:
        Propagates descriptive errors from each sub-module.
    """
    logger.info("Starting ingestion: %s", pdf_path)

    docs = load_pdf_pages(pdf_path)
    logger.info("Loaded %d pages", len(docs))

    if use_ocr_if_needed and needs_ocr(docs):
        logger.info("Text content sparse — falling back to OCR")
        docs = ocr_pdf_pages(pdf_path)

    docs = add_page_descriptions(docs)

    chunks = semantic_like_chunk_docs(docs)
    logger.info("Created %d chunks", len(chunks))

    embeddings = get_embeddings()
    vectorstore = build_faiss(chunks, embeddings)
    save_faiss(vectorstore, index_dir)

    # Write a marker file so the UI knows which PDF is currently indexed.
    try:
        marker = Path(index_dir) / "indexed_file.txt"
        marker.write_text(Path(pdf_path).name, encoding="utf-8")
    except OSError:
        pass  # Non-critical — UI warning will not be shown, but ingestion succeeded.

    result = {
        "pages": len(docs),
        "chunks": len(chunks),
        "index_dir": index_dir,
    }
    logger.info("Ingestion complete: %s", result)
    return result

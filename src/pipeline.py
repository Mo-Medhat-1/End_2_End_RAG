from src.loaders import load_pdf_pages
from src.ocr import ocr_pdf_pages, needs_ocr
from src.metadata import add_page_descriptions
from src.chunking import semantic_like_chunk_docs
from src.embeddings import get_embeddings
from src.vectorstore import build_faiss, save_faiss


def ingest_pdf_to_faiss(pdf_path: str, index_dir: str = "vectorstore/faiss_index", use_ocr_if_needed: bool = True):
    docs = load_pdf_pages(pdf_path)

    if use_ocr_if_needed and needs_ocr(docs):
        docs = ocr_pdf_pages(pdf_path)

    docs = add_page_descriptions(docs)
    chunks = semantic_like_chunk_docs(docs)

    embeddings = get_embeddings()
    vectorstore = build_faiss(chunks, embeddings)
    save_faiss(vectorstore, index_dir)

    return {
        "pages": len(docs),
        "chunks": len(chunks),
        "index_dir": index_dir,
    }

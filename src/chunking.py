"""Text chunking strategies for document processing."""
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def semantic_like_chunk_docs(
    docs: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    Split documents into overlapping chunks using recursive character splitting.

    Separators are ordered by semantic strength:
    paragraphs → newlines → sentences → words → characters.
    Each chunk preserves the original page metadata and receives a unique chunk_id.

    Args:
        docs:          List of LangChain Documents to chunk.
        chunk_size:    Maximum character count per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of chunk Documents. Empty or whitespace-only chunks are discarded.

    Raises:
        ValueError: If the input document list is empty.
        ValueError: If all chunks after splitting are empty/whitespace.
    """
    if not docs:
        raise ValueError(
            "Cannot chunk an empty document list. "
            "Ensure the PDF was loaded and parsed successfully."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "؟ ", "? ", "! ", " ", ""],
    )

    raw_chunks = splitter.split_documents(docs)

    # Discard empty / whitespace-only chunks that pollute retrieval results.
    chunks = [ch for ch in raw_chunks if ch.page_content.strip()]

    if not chunks:
        raise ValueError(
            "All chunks after splitting are empty. "
            "The PDF may contain only images or unreadable content. "
            "Try enabling OCR."
        )

    for i, ch in enumerate(chunks):
        source = ch.metadata.get("source", "doc")
        page = ch.metadata.get("page_number", "x")
        ch.metadata["chunk_id"] = f"{source}_p{page}_c{i}"
        ch.metadata["chunk_description"] = " ".join(ch.page_content.split()[:20])

    logger.info(
        "Produced %d chunks from %d pages (discarded %d empty)",
        len(chunks), len(docs), len(raw_chunks) - len(chunks),
    )
    return chunks

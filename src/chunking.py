"""Text chunking strategies for document processing."""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def semantic_like_chunk_docs(
    docs: List[Document], chunk_size: int = 700, chunk_overlap: int = 120
) -> List[Document]:
    """
    Split documents into overlapping chunks using recursive character splitting.

    Separators are ordered by semantic strength:
    paragraphs → newlines → sentences → words.
    Each chunk preserves the original page metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "؟ ", "? ", "! ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    for i, ch in enumerate(chunks):
        ch.metadata["chunk_id"] = (
            f"{ch.metadata.get('source','doc')}_p{ch.metadata.get('page_number','x')}_c{i}"
        )
        ch.metadata["chunk_description"] = " ".join(ch.page_content.split()[:20])
    return chunks

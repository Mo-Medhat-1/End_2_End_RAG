from langchain_core.documents import Document
from src.chunking import semantic_like_chunk_docs


def test_chunking_keeps_metadata():
    docs = [Document(page_content="A " * 1000, metadata={"source": "x.pdf", "page_number": 1})]
    chunks = semantic_like_chunk_docs(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "x.pdf"
    assert "chunk_id" in chunks[0].metadata

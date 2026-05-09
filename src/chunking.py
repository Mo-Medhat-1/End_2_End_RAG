from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def semantic_like_chunk_docs(
    docs: List[Document], chunk_size: int = 700, chunk_overlap: int = 120
) -> List[Document]:
    """
    Educational semantic-like chunking:
    - Separates by paragraphs first
    - Then sentences/new lines
    - Preserves page metadata
    For stronger true SemanticChunker, use LangChain Experimental + embeddings.
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

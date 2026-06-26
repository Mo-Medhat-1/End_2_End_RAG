"""RAG chain — retrieval-augmented generation with source attribution."""
import logging
from typing import List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

#: Message returned when the retriever finds no relevant chunks.
_NO_CONTEXT_ANSWER = (
    "I couldn't find any relevant content in the document to answer this question. "
    "Try rephrasing your question or verifying that the correct PDF was indexed."
)


def _safe(value: object) -> str:
    """Return a string representation of *value*, replacing None/empty with 'unknown'."""
    text = str(value).strip() if value is not None else ""
    return text if text and text.lower() != "none" else "unknown"


def format_context(docs: List[Document]) -> str:
    """Format retrieved documents into a numbered, structured context block."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        blocks.append(
            f"[Source {i}] "
            f"file={_safe(meta.get('source'))} | "
            f"page={_safe(meta.get('page_number'))} | "
            f"description={_safe(meta.get('page_description'))}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, docs: List[Document]) -> str:
    """Build a structured RAG prompt with numbered context blocks."""
    context = format_context(docs)
    return (
        "You MUST answer using ONLY the context below. "
        "Do NOT use any outside knowledge.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Instructions:\n"
        "- Give a clear, direct answer.\n"
        "- If the answer is not in the context, say exactly: "
        "\"I don't know based on the document.\"\n"
        "- Cite the source file and page number.\n\n"
        "Format your response as:\n"
        "**Direct Answer:** <your answer>\n\n"
        "**Evidence:** <supporting text from context>\n\n"
        "**Sources:** <file name + page number>"
    )


def answer_question(
    llm,
    retriever,
    question: str,
) -> Tuple[str, List[Document]]:
    """
    Answer a question using retrieval-augmented generation.

    Args:
        llm:       An object with an ``invoke(prompt: str) -> str`` method.
        retriever: A LangChain retriever with an ``invoke(query: str)`` method.
        question:  The user's natural-language question.

    Returns:
        Tuple of (answer_text, retrieved_documents).

    Raises:
        ValueError: If the question is empty.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    docs: List[Document] = retriever.invoke(question)

    if not docs:
        logger.warning("Retriever returned 0 documents for query: %r", question)
        return _NO_CONTEXT_ANSWER, []

    logger.info("Retrieved %d documents for query: %r", len(docs), question)
    prompt = build_prompt(question, docs)
    answer = llm.invoke(prompt)
    return answer, docs

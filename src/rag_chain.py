"""RAG chain — retrieval-augmented generation with source attribution."""
from typing import List, Tuple
from langchain_core.documents import Document


def format_context(docs: List[Document]) -> str:
    """Format retrieved documents into a structured context string."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata
        blocks.append(
            f"[Source {i}] file={meta.get('source')} | "
            f"page={meta.get('page_number')} | "
            f"description={meta.get('page_description')}\n"
            f"{d.page_content}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, docs: List[Document]) -> str:
    """Build a structured RAG prompt with context, question, and formatting instructions."""
    context = format_context(docs)

    return f"""You MUST answer using the context below.

Context:
{context}

Question:
{question}

Instructions:
- Give a clear, direct answer.
- If the answer is not found, say: "I don't know based on the document."
- Show supporting evidence from the context.
- List sources with page numbers.

Format:

**Direct Answer:**
<your answer>

**Evidence:**
<supporting text from context>

**Sources:**
<file name + page number>
"""


def answer_question(
    llm, retriever, question: str
) -> Tuple[str, List[Document]]:
    """
    Answer a question using retrieval-augmented generation.

    Returns a tuple of (answer_text, retrieved_documents).
    """
    docs = retriever.invoke(question)
    prompt = build_prompt(question, docs)
    answer = llm.invoke(prompt)
    return answer, docs

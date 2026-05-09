from typing import Dict, List, Tuple
from langchain_core.documents import Document


def format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata
        blocks.append(
            f"[Source {i}] file={meta.get('source')} | page={meta.get('page_number')} | "
            f"description={meta.get('page_description')}\n{d.page_content}"
        )
    return "\n\n".join(blocks)


def build_prompt(question, docs):
    context = format_context(docs)

    return f"""
        You MUST answer using the context below.

        Context:
        {context}

        Question:
        {question}

        Instructions:
        - Give a clear direct answer.
        - If the answer is not found, say: "I don't know based on the document".
        - Then show supporting evidence.
        - Then list sources with page numbers.

        Format your answer like this:

        Direct Answer:
        \\n\n
        - <your answer here>

        Evidence:
        \\n\\n
        - <supporting text>

        Sources:
        - file name + page number
        """


def answer_question(llm, retriever, question: str) -> Tuple[str, List[Document]]:
    docs = retriever.invoke(question)
    prompt = build_prompt(question, docs)
    answer = llm.invoke(prompt)
    return answer, docs

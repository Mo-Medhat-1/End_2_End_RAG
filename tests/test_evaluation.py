from langchain_core.documents import Document
from src.evaluation import recall_at_k


def test_recall_at_k():
    docs = [Document(page_content="x", metadata={"page_number": 3})]
    assert recall_at_k(docs, 3) == 1
    assert recall_at_k(docs, 4) == 0

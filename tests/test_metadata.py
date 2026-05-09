from langchain_core.documents import Document
from src.metadata import add_page_descriptions


def test_add_page_descriptions():
    docs = [Document(page_content="Annual leave policy is 21 days.", metadata={"page_number": 1})]
    out = add_page_descriptions(docs)
    assert "page_description" in out[0].metadata
    assert "Annual" in out[0].metadata["page_description"]

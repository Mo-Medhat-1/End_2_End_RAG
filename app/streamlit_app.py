import sys
from pathlib import Path
import streamlit as st

from src.pipeline import ingest_pdf_to_faiss
from src.embeddings import get_embeddings
from src.vectorstore import load_faiss
from src.llm_qwen import QwenLLM
from src.rag_chain import answer_question

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "vectorstore" / "faiss_index"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Qwen PDF RAG", layout="wide")
st.title("📄 End-to-End PDF RAG with Qwen + FAISS")

st.markdown("""
Upload a PDF, build a FAISS index, ask questions, and inspect retrieved sources with page numbers.
""")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-K retrieved chunks", 1, 10, 5)
    use_ocr = st.checkbox("Use OCR if PDF text is weak", value=True)
    model_name = st.text_input("Qwen model", value="Qwen/Qwen2.5-0.5B-Instruct")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    # raw_dir = ROOT / "data" / "raw"
    # raw_dir.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_RAW_DIR / uploaded.name
    raw_path.write_bytes(uploaded.getbuffer())

    if st.button("Build / Rebuild Index"):
        with st.spinner(
            "Reading PDF, extracting text/OCR, chunking, embedding, and saving FAISS..."
        ):
            stats = ingest_pdf_to_faiss(
                str(raw_path), INDEX_DIR, use_ocr_if_needed=use_ocr
            )
        st.success(f"Index ready: {stats['pages']} pages, {stats['chunks']} chunks")

question = st.text_input("Ask a question about the uploaded PDF")

if question:
    with st.spinner("Loading FAISS and Qwen..."):
        embeddings = get_embeddings()
        vectorstore = load_faiss(INDEX_DIR, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        llm = QwenLLM(model_name=model_name)

    answer, docs = answer_question(llm, retriever, question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved Sources")
    for i, d in enumerate(docs, 1):
        with st.expander(
            f"Source {i}: {d.metadata.get('source')} | page {d.metadata.get('page_number')}"
        ):
            st.write("Description:", d.metadata.get("page_description"))
            st.write(d.page_content[:1500])

    st.subheader("Human Feedback")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("👍 Useful")
    with col2:
        st.button("👎 Not useful")
    with col3:
        st.slider("Rating", 1, 5, 3)

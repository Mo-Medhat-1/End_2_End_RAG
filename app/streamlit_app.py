"""Streamlit UI for the End-to-End PDF RAG system."""
import sys
from pathlib import Path

import streamlit as st

# ── Ensure project root is on sys.path ──
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ingest_pdf_to_faiss  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.vectorstore import load_faiss  # noqa: E402
from src.llm_qwen import QwenLLM  # noqa: E402
from src.rag_chain import answer_question  # noqa: E402

# ── Paths ──
DATA_RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "vectorstore" / "faiss_index"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ── Cached resources (loaded ONCE, not on every rerun) ──
@st.cache_resource
def get_llm(model_name: str) -> QwenLLM:
    """Load the Qwen LLM via HF Inference API (cached)."""
    return QwenLLM(model_name=model_name)


@st.cache_resource
def get_cached_embeddings():
    """Load the embedding model (cached)."""
    return get_embeddings()


# ── Page config ──
st.set_page_config(page_title="Qwen PDF RAG", layout="wide")
st.title("📄 End-to-End PDF RAG with Qwen + FAISS")
st.caption(
    "Upload a PDF, build a FAISS index, ask questions, "
    "and inspect retrieved sources with page numbers."
)

# ── Sidebar settings ──
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Top-K retrieved chunks", 1, 10, 5)
    use_ocr = st.checkbox("Use OCR if PDF text is weak", value=True)
    model_name = st.text_input(
        "Qwen model", value="Qwen/Qwen2.5-0.5B-Instruct"
    )

# ── Check currently indexed file ──
indexed_file_marker = INDEX_DIR / "indexed_file.txt"
current_indexed_name = ""
if indexed_file_marker.exists():
    try:
        current_indexed_name = indexed_file_marker.read_text(encoding="utf-8").strip()
    except Exception:
        pass

# ── PDF upload ──
uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    raw_path = DATA_RAW_DIR / uploaded.name
    raw_path.write_bytes(uploaded.getbuffer())

    if st.button("🔨 Build / Rebuild Index"):
        with st.spinner("Reading PDF, chunking, embedding, saving FAISS..."):
            stats = ingest_pdf_to_faiss(
                str(raw_path), str(INDEX_DIR), use_ocr_if_needed=use_ocr
            )
            try:
                indexed_file_marker.write_text(uploaded.name, encoding="utf-8")
                current_indexed_name = uploaded.name
            except Exception:
                pass
        st.success(
            f"✅ Index ready — {stats['pages']} pages, {stats['chunks']} chunks"
        )
        st.rerun()

    if current_indexed_name != uploaded.name:
        st.warning(
            f"⚠️ **Attention:** The uploaded file `{uploaded.name}` has not been indexed yet. "
            f"The system is still querying the previously indexed file `{current_indexed_name or 'None'}`. "
            "Please click the **Build / Rebuild Index** button above to index your new file."
        )


# ── Question answering ──
question = st.text_input("Ask a question about the uploaded PDF")

if question:
    # Check that an index exists
    index_file = INDEX_DIR / "index.faiss"
    if not index_file.exists():
        st.warning("⚠️ Please upload a PDF and click 'Build Index' first.")
    else:
        with st.spinner("Querying..."):
            try:
                embeddings = get_cached_embeddings()
                vectorstore = load_faiss(str(INDEX_DIR), embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": k})
                llm = get_llm(model_name)
                answer, docs = answer_question(llm, retriever, question)

                st.subheader("💬 Answer")
                st.write(answer)

                st.subheader("📚 Retrieved Sources")
                for i, d in enumerate(docs, 1):
                    page = d.metadata.get("page_number", "?")
                    source = d.metadata.get("source", "unknown")
                    with st.expander(f"Source {i}: {source} — page {page}"):
                        desc = d.metadata.get("page_description", "")
                        if desc:
                            st.write("**Description:**", desc)
                        st.write(d.page_content[:1500])

                st.subheader("👍 Feedback")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("👍 Useful")
                with col2:
                    st.button("👎 Not useful")
                with col3:
                    st.slider("Rating", 1, 5, 3)
            except Exception as e:
                err_str = str(e)
                if "401" in err_str or "Unauthorized" in err_str or "username or password" in err_str:
                    st.error(
                        "🔴 **Authentication Error (401):** Your Hugging Face token is invalid or has been revoked.\n\n"
                        "This usually happens because the token was committed to Git/GitHub, causing Hugging Face to automatically revoke it for safety.\n\n"
                        "**To fix this:**\n"
                        "1. Go to your [Hugging Face Access Tokens page](https://huggingface.co/settings/tokens).\n"
                        "2. Create a new token (with read permission).\n"
                        "3. Replace the token in your `.env` file:\n"
                        "```env\n"
                        "huggingface_token=your_new_token_here\n"
                        "```\n"
                        "4. Re-run your query."
                    )
                else:
                    st.error(f"🔴 **An error occurred:** {e}")


"""Streamlit UI for the End-to-End PDF RAG system."""
import os
import sys
import logging
from pathlib import Path

import streamlit as st

# ── Ensure project root is on sys.path (works locally and on Streamlit Cloud) ──
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Inject Streamlit Cloud secrets into environment variables ──
# This must happen BEFORE any src module is imported so that get_token()
# inside llm_qwen.py can read them from os.getenv() as a fallback.
try:
    for _key in ("huggingface_token", "HF_TOKEN", "QWEN_MODEL", "EMBEDDING_MODEL"):
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # st.secrets not available (e.g. running locally without secrets)

from src.pipeline import ingest_pdf_to_faiss  # noqa: E402
from src.embeddings import get_embeddings      # noqa: E402
from src.vectorstore import load_faiss         # noqa: E402
from src.llm_qwen import QwenLLM              # noqa: E402
from src.rag_chain import answer_question      # noqa: E402

logging.basicConfig(level=logging.INFO)

# ── Paths ──
DATA_RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR    = ROOT / "vectorstore" / "faiss_index"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Cached resources — loaded ONCE per session
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_cached_embeddings():
    """Download and cache the sentence-transformer embedding model."""
    try:
        return get_embeddings()
    except RuntimeError as exc:
        st.error(f"⚠️ **Embedding model error:** {exc}")
        st.stop()


@st.cache_resource(show_spinner="Connecting to Hugging Face API…")
def get_llm(model_name: str) -> QwenLLM:
    """Instantiate and cache the Qwen LLM (one instance per model name)."""
    try:
        return QwenLLM(model_name=model_name)
    except ValueError as exc:
        # Missing token — show clear instructions and stop the app.
        st.error(
            "🔴 **Hugging Face token not configured.**\n\n"
            f"{exc}\n\n"
            "**For Streamlit Cloud:** go to *App Settings → Secrets* and add:\n"
            "```toml\n"
            'huggingface_token = "hf_your_token_here"\n'
            "```"
        )
        st.stop()


# ─────────────────────────────────────────────
# Helper — classify exceptions into user messages
# ─────────────────────────────────────────────
def _user_friendly_error(exc: Exception) -> str:
    msg = str(exc)
    if "401" in msg or "Unauthorized" in msg or "username or password" in msg:
        return (
            "🔴 **Authentication Error (401)** — Your Hugging Face token is invalid "
            "or has been revoked (this happens when a token is accidentally pushed to GitHub).\n\n"
            "**Fix:** Generate a new token at "
            "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) "
            "and update your Secrets."
        )
    if "429" in msg or "rate limit" in msg.lower():
        return (
            "⏳ **Rate Limit (429)** — The Hugging Face free tier is temporarily busy. "
            "Please wait 30 seconds and try again."
        )
    if "503" in msg or "unavailable" in msg.lower():
        return (
            "🔴 **Service Unavailable (503)** — Hugging Face inference servers are "
            "temporarily down. Try again in a few minutes."
        )
    if "tesseract" in msg.lower():
        return (
            "🔴 **Tesseract OCR not found** — The OCR engine is missing.\n\n"
            "This is configured via `packages.txt` on Streamlit Cloud. "
            "If you see this locally, install Tesseract from "
            "https://github.com/UB-Mannheim/tesseract/wiki"
        )
    if "token" in msg.lower() and "not found" in msg.lower():
        return f"🔴 **Missing API Token** — {exc}"
    if "FileNotFoundError" in type(exc).__name__ or "not found" in msg.lower():
        return f"📁 **File Error** — {exc}"
    if "corrupted" in msg.lower() or "password" in msg.lower():
        return f"📄 **PDF Error** — {exc}"
    if "empty" in msg.lower() or "no pages" in msg.lower():
        return f"📄 **Empty PDF** — {exc}"
    return f"🔴 **Unexpected error:** {exc}"


# ─────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG — Qwen + FAISS",
    page_icon="📄",
    layout="wide",
)

st.title("📄 End-to-End PDF RAG with Qwen + FAISS")
st.caption(
    "Upload a PDF → build a FAISS index → ask questions "
    "→ get answers with source page numbers."
)

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Top-K retrieved chunks", min_value=1, max_value=10, value=5, key="top_k")
    use_ocr = st.checkbox("Use OCR if PDF text is sparse", value=True, key="use_ocr")
    model_name = st.text_input(
        "Qwen model",
        value=os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        key="model_name",
    )
    st.divider()
    st.caption("🔒 API keys are read from Streamlit Secrets — never stored in the repo.")

# ── Read the currently-indexed file marker ──
indexed_file_marker = INDEX_DIR / "indexed_file.txt"
current_indexed_name = ""
if indexed_file_marker.exists():
    try:
        current_indexed_name = indexed_file_marker.read_text(encoding="utf-8").strip()
    except OSError:
        pass

# ─────────────────────────────────────────────
# PDF Upload & Indexing
# ─────────────────────────────────────────────
uploaded = st.file_uploader("📂 Upload a PDF file", type=["pdf"], key="pdf_uploader")

if uploaded:
    raw_path = DATA_RAW_DIR / uploaded.name
    try:
        raw_path.write_bytes(uploaded.getbuffer())
    except OSError as exc:
        st.error(f"⚠️ Could not save uploaded file: {exc}")
        st.stop()

    if current_indexed_name and current_indexed_name != uploaded.name:
        st.warning(
            f"⚠️ **Note:** Currently indexed file is `{current_indexed_name}`. "
            f"Click **Build / Rebuild Index** to switch to `{uploaded.name}`."
        )

    if st.button("🔨 Build / Rebuild Index", key="build_btn"):
        with st.spinner("Reading PDF → chunking → embedding → saving FAISS index…"):
            try:
                stats = ingest_pdf_to_faiss(
                    str(raw_path),
                    str(INDEX_DIR),
                    use_ocr_if_needed=use_ocr,
                )
                current_indexed_name = uploaded.name
                st.success(
                    f"✅ Index ready — **{stats['pages']}** pages, "
                    f"**{stats['chunks']}** chunks indexed from `{uploaded.name}`."
                )
                st.rerun()
            except Exception as exc:
                st.error(_user_friendly_error(exc))

# ─────────────────────────────────────────────
# Question Answering
# ─────────────────────────────────────────────
st.divider()
question = st.text_input(
    "💬 Ask a question about the uploaded PDF",
    placeholder="e.g. What is the main topic of this document?",
    key="question_input",
)

if question:
    index_file = INDEX_DIR / "index.faiss"
    if not index_file.exists():
        st.warning(
            "⚠️ No index found. Please upload a PDF and click **Build / Rebuild Index** first."
        )
    else:
        with st.spinner("Searching the document and generating an answer…"):
            try:
                embeddings   = get_cached_embeddings()
                vectorstore  = load_faiss(str(INDEX_DIR), embeddings)
                retriever    = vectorstore.as_retriever(search_kwargs={"k": k})
                llm          = get_llm(model_name)
                answer, docs = answer_question(llm, retriever, question)

            except Exception as exc:
                st.error(_user_friendly_error(exc))
                st.stop()

        # ── Answer ──
        st.subheader("💬 Answer")
        st.markdown(answer)

        # ── Retrieved Sources ──
        if docs:
            st.subheader("📚 Retrieved Sources")
            for i, doc in enumerate(docs, 1):
                page   = doc.metadata.get("page_number", "?")
                source = doc.metadata.get("source", "unknown")
                with st.expander(f"Source {i}: `{source}` — page {page}", expanded=(i == 1)):
                    desc = doc.metadata.get("page_description", "")
                    if desc:
                        st.caption(f"**Page summary:** {desc}")
                    st.write(doc.page_content[:1500])

        # ── Feedback ──
        st.subheader("👍 Was this answer helpful?")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.button("👍 Useful", key=f"btn_useful_{hash(question)}")
        with col2:
            st.button("👎 Not useful", key=f"btn_not_useful_{hash(question)}")
        with col3:
            st.slider("Rate the answer", 1, 5, 3, key=f"slider_rating_{hash(question)}")

"""Streamlit UI for the End-to-End PDF RAG system."""
import os
import sys
import logging
from pathlib import Path

import streamlit as st

# ── Project root on sys.path (local + Streamlit Cloud) ──────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Inject Streamlit Cloud Secrets → os.environ BEFORE any src import ────────
try:
    for _k in ("huggingface_token", "HF_TOKEN", "QWEN_MODEL", "EMBEDDING_MODEL"):
        if _k in st.secrets and _k not in os.environ:
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

from src.pipeline import ingest_pdf_to_faiss  # noqa: E402
from src.embeddings import get_embeddings      # noqa: E402
from src.vectorstore import load_faiss         # noqa: E402
from src.llm_qwen import QwenLLM              # noqa: E402
from src.rag_chain import answer_question      # noqa: E402

logging.basicConfig(level=logging.INFO)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR    = ROOT / "vectorstore" / "faiss_index"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="PDF Intelligence — RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Header banner ── */
.rag-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2.2rem 2.5rem 1.8rem;
    border-radius: 18px;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 40px rgba(0,0,0,.45);
}
.rag-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 .35rem;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.rag-header p {
    color: rgba(255,255,255,.55);
    font-size: .95rem;
    margin: 0;
    font-weight: 400;
}

/* ── Metric cards ── */
.metric-grid { display:flex; gap:1rem; margin:1rem 0 1.5rem; }
.metric-card {
    flex: 1;
    background: rgba(96,165,250,.08);
    border: 1px solid rgba(96,165,250,.2);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    text-align: center;
}
.metric-card .val {
    font-size: 1.9rem;
    font-weight: 700;
    color: #60a5fa;
    line-height: 1;
}
.metric-card .lbl {
    font-size: .75rem;
    color: rgba(255,255,255,.45);
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-top: .35rem;
}

/* ── Answer box ── */
.answer-card {
    background: linear-gradient(135deg,rgba(15,12,41,.9),rgba(48,43,99,.7));
    border: 1px solid rgba(167,139,250,.25);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin: .8rem 0 1.4rem;
    box-shadow: 0 4px 24px rgba(0,0,0,.25);
}

/* ── Source badge ── */
.page-badge {
    display: inline-block;
    background: rgba(167,139,250,.18);
    color: #a78bfa;
    border: 1px solid rgba(167,139,250,.35);
    border-radius: 20px;
    padding: .15rem .75rem;
    font-size: .78rem;
    font-weight: 600;
    margin-bottom: .5rem;
}

/* ── Step label ── */
.step-tag {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    font-size: .8rem;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: .6rem;
}

/* ── Sidebar ── */
.sidebar-section {
    background: rgba(255,255,255,.04);
    border-radius: 12px;
    padding: .9rem 1rem;
    margin-bottom: .8rem;
}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_cached_embeddings():
    try:
        return get_embeddings()
    except RuntimeError as exc:
        st.error(f"⚠️ **Embedding model failed to load:** {exc}")
        st.stop()


@st.cache_resource(show_spinner="Connecting to Hugging Face API…")
def get_llm(model_name: str) -> QwenLLM:
    try:
        return QwenLLM(model_name=model_name)
    except ValueError as exc:
        st.error(
            "🔴 **Hugging Face token not configured.**\n\n"
            f"{exc}\n\n"
            "**Streamlit Cloud:** go to *App Settings → Secrets* and add:\n"
            "```toml\nhuggingface_token = \"hf_your_token_here\"\n```"
        )
        st.stop()


# ── Error classifier ─────────────────────────────────────────────────────────
def _friendly_error(exc: Exception) -> str:
    msg = str(exc)
    if "401" in msg or "Unauthorized" in msg or "username or password" in msg:
        return (
            "🔴 **Authentication Error (401)** — Your HF token is invalid or revoked.\n\n"
            "Generate a fresh one at [huggingface.co/settings/tokens]"
            "(https://huggingface.co/settings/tokens) and update your Secrets."
        )
    if "429" in msg or "rate limit" in msg.lower():
        return "⏳ **Rate limit (429)** — HF free tier is busy. Wait 30 s and try again."
    if "503" in msg or "unavailable" in msg.lower():
        return "🔴 **Service unavailable (503)** — HF servers are down. Try in a few minutes."
    if "tesseract" in msg.lower():
        return (
            "🔴 **Tesseract not found** — OCR engine missing.\n"
            "Ensure `tesseract-ocr` is listed in `packages.txt` for Streamlit Cloud."
        )
    if "password" in msg.lower():
        return "🔐 **Encrypted PDF** — Remove the password protection and re-upload."
    if "corrupted" in msg.lower() or "invalid pdf" in msg.lower():
        return "📄 **Corrupted PDF** — The file could not be read. Try re-exporting it."
    if "empty" in msg.lower() or "no pages" in msg.lower() or "zero pages" in msg.lower():
        return "📄 **Empty PDF** — No text or images were found in this file."
    if "no chunks" in msg.lower() or "all chunks" in msg.lower():
        return (
            "📄 **No indexable content** — The PDF appears to contain only images. "
            "Enable OCR in the sidebar and rebuild the index."
        )
    return f"🔴 **Error:** {exc}"


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
  <h1>🧠 PDF Intelligence</h1>
  <p>Upload any PDF → build a semantic index → ask questions in plain language → get precise answers with page citations.</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    with st.container():
        k = st.slider(
            "Top-K retrieved chunks",
            min_value=1, max_value=10, value=5, key="top_k",
            help="Number of document chunks retrieved per query. Higher = more context, but slower.",
        )
        use_ocr = st.toggle(
            "Enable OCR for scanned PDFs",
            value=True, key="use_ocr",
            help="Run Tesseract OCR when the PDF text is too sparse (image-based / scanned).",
        )

    st.divider()

    st.markdown("### 🤖 Model")
    model_name = st.text_input(
        "Hugging Face model ID",
        value=os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        key="model_name",
        help="Any HF chat/instruct model accessible via the Inference API.",
    )

    st.divider()
    st.caption(
        "🔒 API keys are read from Streamlit Secrets — never stored in the repository."
    )
    st.caption("📦 [View source on GitHub](https://github.com/Mo-Medhat-1/End_2_End_RAG)")


# ── Read indexed-file marker ─────────────────────────────────────────────────
indexed_marker   = INDEX_DIR / "indexed_file.txt"
current_indexed  = ""
if indexed_marker.exists():
    try:
        current_indexed = indexed_marker.read_text(encoding="utf-8").strip()
    except OSError:
        pass


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload & Index
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-tag">📂 Step 1 — Upload & Index</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop a PDF here or click Browse",
    type=["pdf"],
    key="pdf_uploader",
    label_visibility="collapsed",
)

if uploaded:
    raw_path = DATA_RAW_DIR / uploaded.name
    try:
        raw_path.write_bytes(uploaded.getbuffer())
    except OSError as exc:
        st.error(f"⚠️ Could not save file: {exc}")
        st.stop()

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        file_mb = uploaded.size / 1_048_576
        st.markdown(
            f"**📄 {uploaded.name}** &nbsp;·&nbsp; `{file_mb:.1f} MB`",
            unsafe_allow_html=True,
        )
        if current_indexed and current_indexed != uploaded.name:
            st.warning(
                f"⚠️ Currently indexed: **{current_indexed}**. "
                "Rebuild to switch to the new file."
            )

    with col_btn:
        build = st.button("🔨 Build Index", key="build_btn", use_container_width=True)

    if build:
        progress_bar = st.progress(0, text="Starting…")
        status       = st.empty()

        try:
            status.info("📖 Reading PDF pages…")
            progress_bar.progress(15, text="Reading PDF…")

            stats = ingest_pdf_to_faiss(
                str(raw_path),
                str(INDEX_DIR),
                use_ocr_if_needed=use_ocr,
            )
            progress_bar.progress(100, text="Done!")
            current_indexed = uploaded.name
            status.empty()
            progress_bar.empty()

            # Metrics row
            st.markdown(
                f"""
                <div class="metric-grid">
                  <div class="metric-card">
                    <div class="val">{stats['pages']}</div>
                    <div class="lbl">Pages</div>
                  </div>
                  <div class="metric-card">
                    <div class="val">{stats['chunks']}</div>
                    <div class="lbl">Chunks indexed</div>
                  </div>
                  <div class="metric-card">
                    <div class="val">FAISS</div>
                    <div class="lbl">Vector store</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.success(f"✅ Index ready for **{uploaded.name}**")
            st.rerun()

        except Exception as exc:
            progress_bar.empty()
            status.empty()
            st.error(_friendly_error(exc))


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Ask a Question
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="step-tag">💬 Step 2 — Ask a Question</div>', unsafe_allow_html=True)

question = st.text_input(
    "Your question",
    placeholder="e.g. Summarize the main topics of this document.",
    key="question_input",
    label_visibility="collapsed",
)

if question:
    index_file = INDEX_DIR / "index.faiss"
    if not index_file.exists():
        st.warning("⚠️ No index found. Please upload a PDF and click **Build Index** first.")
    else:
        with st.spinner("Searching document and generating answer…"):
            try:
                embeddings   = get_cached_embeddings()
                vectorstore  = load_faiss(str(INDEX_DIR), embeddings)
                retriever    = vectorstore.as_retriever(search_kwargs={"k": k})
                llm          = get_llm(model_name)
                answer, docs = answer_question(llm, retriever, question)
            except Exception as exc:
                st.error(_friendly_error(exc))
                st.stop()

        # ── Answer ────────────────────────────────────────────────────────
        st.markdown("#### 💡 Answer")
        st.markdown(
            f'<div class="answer-card">{answer}</div>',
            unsafe_allow_html=True,
        )

        # ── Sources ───────────────────────────────────────────────────────
        if docs:
            st.markdown(f"#### 📚 Retrieved Sources &nbsp; `{len(docs)} chunks`")
            for i, doc in enumerate(docs, 1):
                page   = doc.metadata.get("page_number", "?")
                source = doc.metadata.get("source", "unknown")
                loader = doc.metadata.get("loader", "")
                desc   = doc.metadata.get("page_description", "")

                with st.expander(
                    f"Source {i} — {source} · page {page}",
                    expanded=(i == 1),
                ):
                    header_cols = st.columns([1, 1, 3])
                    with header_cols[0]:
                        st.markdown(
                            f'<span class="page-badge">Page {page}</span>',
                            unsafe_allow_html=True,
                        )
                    with header_cols[1]:
                        if loader:
                            st.markdown(
                                f'<span class="page-badge">{loader}</span>',
                                unsafe_allow_html=True,
                            )
                    if desc:
                        st.caption(f"📝 {desc}")
                    st.markdown("---")
                    st.write(doc.page_content[:2000])

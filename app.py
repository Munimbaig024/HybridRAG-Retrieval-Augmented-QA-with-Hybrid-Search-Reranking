"""
HybridRAG demo — Streamlit app for Streamlit Community Cloud (free tier).

Wire the TODO-marked functions to your existing src/ modules (ingest.py, query.py).
"""

import json
import os
import time

import streamlit as st

# --- Config -----------------------------------------------------------------

try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    # Fallback to os.environ for local testing if secrets.toml doesn't exist
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

MAX_QUERIES_PER_SESSION = 20

EVAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval", "results.json")


@st.cache_data
def load_eval_stats():
    """Load precomputed eval metrics for the stats panel. Falls back gracefully if missing."""
    try:
        with open(EVAL_RESULTS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "precision_at_5": None,
            "recall_at_5": None,
            "mrr": None,
            "ndcg_at_10": None,
            "faithfulness": None,
        }


def format_eval_stats(stats: dict) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"

    return (
        f"**Precision@5**: {fmt(stats.get('precision_at_5'))}  \n"
        f"**Recall@5**: {fmt(stats.get('recall_at_5'))}  \n"
        f"**MRR**: {fmt(stats.get('mrr'))}  \n"
        f"**nDCG@10**: {fmt(stats.get('ndcg_at_10'))}  \n"
        f"**Faithfulness**: {fmt(stats.get('faithfulness'))}"
    )


# --- Pipeline hooks (wire these to src/) -------------------------------------
# Cache loaded models/index once per app lifetime, not per query/session.

@st.cache_resource
def load_pipeline():
    """Loads and initializes the pipeline from the prebuilt index on disk."""
    from src.pipeline import HybridRAGPipeline
    pipeline = HybridRAGPipeline()
    pipeline.load_state()
    return pipeline


def get_active_pipeline():
    """Returns the session pipeline if docs are uploaded, else the default."""
    if "uploaded_pipeline" in st.session_state:
        return st.session_state.uploaded_pipeline
    return load_pipeline()


def run_dense_only(query: str, k: int = 5):
    """Calls the dense-only retrieval + LLM answer path."""
    pipeline = get_active_pipeline()
    result = pipeline.run_dense_only(query, final_k=k)
    chunks = [c["text"] for c in result["context"]]
    return result["answer"], chunks


def run_hybrid_rerank(query: str, k: int = 5):
    """Calls the BM25 + dense + RRF fusion + cross-encoder rerank + LLM answer path."""
    pipeline = get_active_pipeline()
    result = pipeline.run(query, final_k=k)
    chunks = [c["text"] for c in result["context"]]
    return result["answer"], chunks


def ingest_uploaded_docs(files):
    """Creates a temporary in-memory pipeline for uploaded docs."""
    if len(files) > 3:
        raise ValueError("Maximum of 3 files allowed to save memory.")
        
    documents_data = []
    for f in files:
        if f.size > 500 * 1024:
            raise ValueError(f"File {f.name} is too large (max 500KB).")
        text = f.read().decode('utf-8', errors='ignore')
        documents_data.append({"title": f.name, "summary": text})
        
    from src.document_processor import process_articles_into_chunks
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP
    chunks = process_articles_into_chunks(documents_data, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    from src.pipeline import HybridRAGPipeline
    pipeline = HybridRAGPipeline(db_dir=None) # Ephemeral
    pipeline.ingest_documents(chunks)
    
    st.session_state.uploaded_pipeline = pipeline


# --- Rate limiting ------------------------------------------------------------

def check_rate_limit() -> bool:
    """Returns True if this browser session is still under the query cap."""
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    st.session_state.query_count += 1
    return st.session_state.query_count <= MAX_QUERIES_PER_SESSION


# --- Page setup ----------------------------------------------------------------

st.set_page_config(page_title="HybridRAG", page_icon="🔎", layout="wide")

st.title("HybridRAG")
st.caption(
    "Retrieval-augmented QA over a Wikipedia subset, combining dense + BM25 "
    "retrieval, reciprocal rank fusion, and cross-encoder reranking."
)

# Trigger pipeline load once (cached) so first-query latency doesn't surprise the visitor.
try:
    load_pipeline()
    PIPELINE_READY = True
except NotImplementedError:
    PIPELINE_READY = False

with st.sidebar:
    st.markdown("### Eval results")
    st.markdown(format_eval_stats(load_eval_stats()))
    st.caption(
        "Computed offline against a labeled qrels set — see `eval/` in the repo "
        "for methodology."
    )
    if not PIPELINE_READY:
        st.warning("Pipeline not wired up yet — see TODOs in app.py.")

tab_ask, tab_compare, tab_upload = st.tabs(
    ["Ask", "Compare: dense-only vs hybrid+rerank", "Upload your own docs (optional)"]
)

with tab_ask:
    query = st.text_input("Question", placeholder="Ask something about the loaded corpus...")
    mode = st.radio("Retrieval mode", ["Hybrid + rerank", "Dense only"], horizontal=True)

    if st.button("Ask", type="primary"):
        if not query.strip():
            st.warning("Enter a question first.")
        elif not check_rate_limit():
            st.error(
                f"Rate limit reached ({MAX_QUERIES_PER_SESSION} queries per session). "
                "Refresh the page to reset, or clone the repo to run your own."
            )
        else:
            start = time.time()
            try:
                if mode == "Hybrid + rerank":
                    answer, chunks = run_hybrid_rerank(query)
                else:
                    answer, chunks = run_dense_only(query)
                elapsed = time.time() - start

                st.markdown("#### Answer")
                st.write(answer)
                st.caption(f"Answered in {elapsed:.2f}s using **{mode}**.")

                with st.expander("Retrieved chunks"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.write(chunk)
                        st.divider()
            except NotImplementedError:
                st.error(
                    "Pipeline not wired up yet — connect run_dense_only / "
                    "run_hybrid_rerank to src/query.py."
                )

with tab_compare:
    compare_query = st.text_input(
        "Question", placeholder="Try a query with a specific term or number", key="compare_q"
    )

    if st.button("Compare", type="primary"):
        if not compare_query.strip():
            st.warning("Enter a question first.")
        elif not check_rate_limit():
            st.error(f"Rate limit reached ({MAX_QUERIES_PER_SESSION} queries per session).")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Dense only")
                try:
                    dense_answer, _ = run_dense_only(compare_query)
                    st.write(dense_answer)
                except NotImplementedError:
                    st.info("Pipeline not wired up yet.")

            with col2:
                st.markdown("#### Hybrid + rerank")
                try:
                    hybrid_answer, _ = run_hybrid_rerank(compare_query)
                    st.write(hybrid_answer)
                except NotImplementedError:
                    st.info("Pipeline not wired up yet.")

with tab_upload:
    st.write("Upload documents to index and query against instead of the default corpus.")
    uploaded_files = st.file_uploader("Documents", accept_multiple_files=True)

    if st.button("Ingest"):
        if not uploaded_files:
            st.warning("Choose at least one file first.")
        else:
            try:
                ingest_uploaded_docs(uploaded_files)
                st.success("Ingested. Switch to the Ask tab to query your documents.")
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")

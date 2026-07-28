import sys
import logging
from pathlib import Path

import streamlit as st

# Ensure backend package is importable
sys.path.append(str(Path(__file__).resolve().parent))

from app.rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("corpusiq_chat")


# ──────────────────────────────────────────────
# Pipeline Initialization (cached — loads once)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CorpusIQ models (embeddings, reranker, Qdrant)...")
def load_pipeline() -> RAGPipeline:
    """Initialize the RAG pipeline once and cache across all Streamlit reruns."""
    logger.info("Initializing RAGPipeline for Streamlit...")
    pipeline = RAGPipeline()
    logger.info("RAGPipeline ready.")
    return pipeline


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CorpusIQ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Source cards */
    .source-card {
        background: #1a1a2e;
        border: 1px solid #2d2d44;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .source-card .score {
        color: #667eea;
        font-weight: 600;
    }
    .source-card .subject-line {
        color: #e0e0e0;
        font-weight: 500;
    }
    .source-card .meta {
        color: #999;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar — Retrieval Controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Retrieval Settings")
    st.markdown("---")

    candidate_k = st.slider(
        "Hybrid retrieval candidates",
        min_value=5, max_value=50, value=20,
        help="Number of candidates from RRF fusion before reranking"
    )

    rerank_top_n = st.slider(
        "Reranked results (Top N)",
        min_value=1, max_value=20, value=5,
        help="Number of chunks kept after cross-encoder reranking"
    )

    enable_thread_expansion = st.checkbox(
        "Thread context expansion",
        value=True,
        help="Inject parent thread root for reply emails"
    )

    st.markdown("---")
    st.markdown("### 🔎 Filters")

    from_address_filter = st.text_input(
        "From address",
        placeholder="e.g. john.doe@enron.com",
        help="Filter by sender email"
    ) or None

    sender_domain_filter = st.text_input(
        "Sender domain",
        placeholder="e.g. enron.com",
        help="Filter by sender domain"
    ) or None

    thread_id_filter = st.text_input(
        "Thread ID",
        placeholder="e.g. thread-abc123...",
        help="Filter by specific thread"
    ) or None

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.75rem;'>"
        "CorpusIQ v0.1.0<br>Hybrid RAG · Cross-Encoder · Thread-Aware"
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 CorpusIQ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enterprise email intelligence — ask questions about the Enron email corpus</div>',
    unsafe_allow_html=True,
)

# Initialize pipeline
pipeline = load_pipeline()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Re-render sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📄 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src.get('subject', 'N/A')}**  \n"
                        f"From: `{src.get('from_address', 'N/A')}` · "
                        f"Score: `{src.get('score', 0)}` · "
                        f"Date: `{src.get('date_str', 'N/A')}`  \n"
                        f"Thread: `{src.get('thread_id', 'N/A')}`"
                    )
                    st.markdown("---")

# Chat input
if user_query := st.chat_input("Ask a question about the Enron emails..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching and synthesizing..."):
            try:
                response = pipeline.query(
                    query_text=user_query,
                    candidate_k=candidate_k,
                    rerank_top_n=rerank_top_n,
                    enable_thread_expansion=enable_thread_expansion,
                    from_address_filter=from_address_filter,
                    thread_id_filter=thread_id_filter,
                    sender_domain_filter=sender_domain_filter,
                )

                # Display answer
                st.markdown(response.answer)

                # Display sources
                if response.sources:
                    with st.expander(f"📄 Sources ({len(response.sources)})"):
                        for src in response.sources:
                            st.markdown(
                                f"**{src.get('subject', 'N/A')}**  \n"
                                f"From: `{src.get('from_address', 'N/A')}` · "
                                f"Score: `{src.get('score', 0)}` · "
                                f"Date: `{src.get('date_str', 'N/A')}`  \n"
                                f"Thread: `{src.get('thread_id', 'N/A')}`"
                            )
                            st.markdown("---")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                })

            except Exception as e:
                error_msg = f"Pipeline error: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg, exc_info=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ {error_msg}",
                    "sources": [],
                })

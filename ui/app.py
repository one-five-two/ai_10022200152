# Name: NWAOGU SOMTOCHUKWU SHARON
# Index Number: 10022200152

"""
Streamlit UI for Ghana RAG Chatbot
------------------------------------
Run: streamlit run ui/app.py
"""

import sys
import logging
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go

# ── Page config must be FIRST streamlit call ─────────────────────────────
st.set_page_config(
    page_title="Ghana Knowledge Assistant",
    page_icon="🇬🇭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports after page config ─────────────────────────────────────────────
from src.retrieval.vector_store import VectorStore
from src.pipeline.rag_pipeline import RAGPipeline
from src.innovation.memory_rag import MemoryRAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = "vector_store"

# ─────────────────────────────────────────────────────────────────────────
# Custom CSS — Ghana-inspired colour scheme (red, gold, green)
# ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #CF1920 0%, #FCD116 50%, #006B3F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #FCD116;
        margin-bottom: 0.5rem;
    }
    .chunk-card {
        background: #1f2937 !important;
        color: #ffffff !important;
        border: 1px solid #374151 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        margin-bottom: 0.8rem !important;
        font-size: 0.85rem !important;
        line-height: 1.6 !important;
    }

    .chunk-card,
    .chunk-card p,
    .chunk-card div,
    .chunk-card span {
        color: #ffffff !important;
    }

    .chunk-score {
        color: #22c55e !important;
        font-weight: 600;
    }
    .source-badge {
        background: #CF1920;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #006B3F, #009A57);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,107,63,0.3);
    }
    .memory-indicator {
        background: #e8f5e9;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: #2e7d32;
        border-left: 3px solid #4CAF50;
    }
    .answer-box {
        background: #1f2937 !important;
        color: #ffffff !important;
        padding: 1.5rem !important;
        border-radius: 14px !important;
        border: 1px solid #FCD116 !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }

    /* Force ALL inner text to be white */
    .answer-box,
    .answer-box p,
    .answer-box div,
    .answer-box span,
    .answer-box strong,
    .answer-box em,
    .answer-box li,
    .answer-box ul,
    .answer-box ol {
        color: #ffffff !important;
    }    
    
            
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "chat_history": [],
        "pipeline": None,
        "memory_pipeline": None,
        "debug_info": {},
        "session_id": "session_" + str(id(st.session_state)),
        "use_memory": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ─────────────────────────────────────────────────────────────────────────
# Load pipeline (cached)
# ─────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_pipeline(top_k: int, use_expansion: bool, prompt_ver: str):
    """Loads and caches the RAG pipeline. Re-loads only if params change."""
    vs_path = Path(VECTOR_STORE_DIR)
    if not vs_path.exists():
        return None, "Vector store not found. Run ingestion first:\n\n`python -m src.pipeline.ingest --csv data/ghana_elections.csv --pdf data/ghana_budget_2025.pdf`"

    store = VectorStore.load(VECTOR_STORE_DIR)
    pipeline = RAGPipeline(
        vector_store=store,
        top_k=top_k,
        use_query_expansion=use_expansion,
        prompt_version=prompt_ver,
    )
    return pipeline, None


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    top_k = st.slider("Retrieved chunks (top-k)", 3, 10, 5)
    use_expansion = st.toggle("Query Expansion", value=True,
                               help="Generates alternative phrasings to improve recall")
    use_memory = st.toggle("Conversation Memory", value=True,
                            help="Maintains context across turns (Part G innovation)")
    prompt_ver = st.selectbox("Prompt template", ["v2", "v1", "v3"],
                               help="v1=basic, v2=grounded (recommended), v3=chain-of-thought")

    st.divider()
    st.markdown("### 📊 About")
    st.caption("**Data sources:**")
    st.caption("• Ghana Election Dataset (CSV)")
    st.caption("• Ghana 2025 Budget (PDF)")
    st.caption("**Model:** Free local FLAN-T5")
    st.caption("**Embeddings:** all-MiniLM-L6-v2")
    st.caption("**Index:** FAISS IndexFlatIP")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        if st.session_state.memory_pipeline:
            st.session_state.memory_pipeline.clear_memory()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🇬🇭 Ghana Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions about Ghana\'s elections and 2025 national budget.</div>', unsafe_allow_html=True)

# Load pipeline
base_pipeline, error_msg = load_pipeline(top_k, use_expansion, prompt_ver)

if error_msg:
    st.error(error_msg)
    st.stop()

# Wrap with memory pipeline if needed
if use_memory:
    if st.session_state.memory_pipeline is None:
        from src.innovation.memory_rag import MemoryRAGPipeline
        st.session_state.memory_pipeline = MemoryRAGPipeline(
            base_pipeline, session_id=st.session_state.session_id
        )
    active_pipeline = st.session_state.memory_pipeline
else:
    active_pipeline = base_pipeline

# ── Sample questions ──────────────────────────────────────────────────────

st.markdown("**💡 Try these questions:**")
sample_cols = st.columns(3)
sample_questions = [
    "What are Ghana's key revenue targets for 2025?",
    "How many parliamentary seats did the NPP win?",
    "What is the education budget allocation for 2025?",
]
for i, q in enumerate(sample_questions):
    if sample_cols[i].button(q, use_container_width=True):
        st.session_state.pending_query = q

# ── Chat history ──────────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Query input ───────────────────────────────────────────────────────────

pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask about Ghana elections or 2025 budget...") or pending

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            result = active_pipeline.query(user_input)

        answer = result["answer"]

        # Show memory indicator if resolution happened
        if result.get("query_resolved_to") and result["query_resolved_to"] != user_input:
            st.markdown(
                f'<div class="memory-indicator">🧠 Interpreted as: <em>"{result["query_resolved_to"]}"</em></div>',
                unsafe_allow_html=True
            )
            st.write("")

        # Answer box
        st.markdown(
            f"""
            <div style="
                background-color:#1f2937 !important;
                color:#ffffff !important;
                padding:24px !important;
                border-radius:14px !important;
                border:1px solid #FCD116 !important;
                font-size:17px !important;
                line-height:1.7 !important;
            ">
                <span style="color:#ffffff !important;">
                    {answer}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Expandable: Retrieved Chunks ─────────────────────────────────
        with st.expander(f"📄 Retrieved Context ({len(result['retrieved_chunks'])} chunks)"):
            for i, chunk in enumerate(result["retrieved_chunks"]):
                source = chunk.get("source", "unknown")
                score = chunk.get("similarity_score", 0)
                page = chunk.get("metadata", {}).get("page", "")

                source_label = "🗳️ Election" if source == "election_csv" else "📊 Budget"
                page_info = f" · Page {page}" if page else ""

                st.markdown(
                    f'<div class="chunk-card">'
                    f'<span class="source-badge">{source_label}{page_info}</span> '
                    f'<span class="chunk-score">Score: {score:.4f}</span><br><br>'
                    f'{chunk["text"][:300]}...'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Similarity score bar chart
            scores = result["similarity_scores"]
            fig = go.Figure(go.Bar(
                x=[f"Chunk {i+1}" for i in range(len(scores))],
                y=scores,
                marker_color=["#006B3F" if s > 0.6 else "#FCD116" if s > 0.4 else "#CF1920" for s in scores],
            ))
            fig.update_layout(
                title="Similarity Scores",
                yaxis_range=[0, 1],
                height=250,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Expandable: Latency breakdown ────────────────────────────────
        with st.expander("⏱️ Performance"):
            lat = result.get("latency_ms", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Retrieval", f"{lat.get('retrieval', 0):.0f}ms")
            c2.metric("Prompt Build", f"{lat.get('prompt_build', 0):.0f}ms")
            c3.metric("LLM", f"{lat.get('llm', 0):.0f}ms")
            c4.metric("Total", f"{lat.get('total', 0):.0f}ms")

        # ── Expandable: Debug prompt ─────────────────────────────────────
        with st.expander("🔧 Debug: Final Prompt"):
            st.code(result.get("final_prompt", ""), language="text")

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

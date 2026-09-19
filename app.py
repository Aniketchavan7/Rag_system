"""
Streamlit Web Interface for the Agentic AI eBook RAG Chatbot.
Provides interactive chat, clickable sample queries, confidence indicators,
and expandable inspection of retrieved context chunks.
"""

import time
import streamlit as st

from config import LLM_MODEL, VECTOR_STORE_TYPE
from store import get_vector_store
from graph import ask_question

# Page Configuration
st.set_page_config(
    page_title="Agentic AI eBook - RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .score-badge-high {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-badge-med {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-badge-low {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .chunk-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sample queries for evaluation
SAMPLE_QUERIES = [
    "what is agentic ai",
    "tell me about multi agent systems",
    "how companies using agentic ai",
    "what are the layers in agentic ai architecture",
    "multi agent sales forecasting example",
    "challenges in multi agent systems",
    "types of agentic ai systems",
    "capabilities of agentic ai",
    "reactive to proactive technology"
]

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_query" not in st.session_state:
    st.session_state.input_query = ""


# Sidebar
with st.sidebar:
    st.title("System Status")
    st.markdown("---")

    # Knowledge Base Stats
    try:
        store = get_vector_store()
        total_chunks = store.count()
    except Exception:
        total_chunks = "Ready"

    st.markdown(f"**Knowledge Base:** `Agentic AI eBook` (60 pages)")
    st.markdown(f"**Vector Store:** `{VECTOR_STORE_TYPE.upper()}` ({total_chunks} chunks)")
    st.markdown(f"**LLM Engine:** `{LLM_MODEL}`")
    st.markdown(f"**Pipeline:** `LangGraph StateGraph`")

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    st.link_button("FastAPI Docs", "http://localhost:8000/docs", use_container_width=True)

    st.markdown("---")
    st.subheader("Sample Questions")
    st.caption("Click any question to try it:")

    for idx, sample in enumerate(SAMPLE_QUERIES):
        if st.button(sample, key=f"sample_{idx}", use_container_width=True):
            st.session_state.selected_sample = sample

    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main View
st.markdown('<div class="main-title">Agentic AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Grounded Q&A over the Agentic AI eBook using LangGraph and ChromaDB</div>', unsafe_allow_html=True)

def render_confidence_badge(score: float, grounded: bool):
    pct = int(score * 100)
    col1, _ = st.columns([2, 8])
    with col1:
        if pct >= 70 and grounded:
            st.markdown(f'<span class="score-badge-high">Confidence: {pct}% (Grounded)</span>', unsafe_allow_html=True)
        elif pct >= 45 and grounded:
            st.markdown(f'<span class="score-badge-med">Confidence: {pct}% (Moderate)</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="score-badge-low">Confidence: {pct}% (Out of Domain)</span>', unsafe_allow_html=True)


def render_retrieved_chunks(chunks: list):
    if not chunks:
        return
    with st.expander(f"Retrieved Chunks ({len(chunks)})"):
        for i, c in enumerate(chunks, 1):
            st.markdown(
                f"**Source #{i} — Page {c.get('page', '?')}** | Similarity Score: `{c.get('score', 0):.3f}`"
            )
            st.info(c.get("text", ""))


# Render Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            render_confidence_badge(msg.get("confidence_score", 0.0), msg.get("is_grounded", False))
            render_retrieved_chunks(msg.get("chunks", []))


# Handle input from sample button or chat input
user_prompt = None
if "selected_sample" in st.session_state and st.session_state.selected_sample:
    user_prompt = st.session_state.selected_sample
    st.session_state.selected_sample = None
else:
    user_prompt = st.chat_input("Ask anything from the Agentic AI eBook...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching eBook knowledge base and verifying grounding..."):
            start_t = time.time()
            result = ask_question(user_prompt)
            duration = round(time.time() - start_t, 2)

            answer = result["final_answer"]
            score = result["confidence_score"]
            grounded = result["is_grounded"]
            chunks = result["retrieved_context_chunks"]

            st.markdown(answer)
            render_confidence_badge(score, grounded)
            render_retrieved_chunks(chunks)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "confidence_score": score,
        "is_grounded": grounded,
        "chunks": chunks
    })

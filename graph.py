# LangGraph RAG pipeline

import os
from typing import List, Dict, Any, Optional, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    TOP_K,
    CONFIDENCE_THRESHOLD,
)
from store import get_vector_store

class RAGState(TypedDict, total=False):
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    context_text: str
    answer: str
    confidence_score: float
    is_grounded: bool
    refusal: bool
    error: bool


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0.0,
        max_tokens=1000
    )


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """Fetch top-k chunks from vector store."""
    question = state["question"].strip()
    store = get_vector_store()

    chunks = store.search(query=question, top_k=TOP_K)

    context_parts = []
    for idx, chunk in enumerate(chunks, 1):
        page = chunk.get("page", "Unknown")
        cid = chunk.get("id", f"chunk_{idx}")
        score = chunk.get("score", 0.0)
        context_parts.append(
            f"[Source {idx} | Page {page} | Score: {score:.3f}]\n{chunk['text']}"
        )

    context_text = "\n\n---\n\n".join(context_parts)
    return {
        "retrieved_chunks": chunks,
        "context_text": context_text
    }


def evaluate_relevance_node(state: RAGState) -> Dict[str, Any]:
    """Check if retrieved chunks are relevant enough to bother calling the LLM."""
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"confidence_score": 0.0, "refusal": True}

    top_score = max(c.get("score", 0.0) for c in chunks)
    avg_score = sum(c.get("score", 0.0) for c in chunks) / len(chunks)

    # 0.45 was picked after testing — lower lets garbage through, higher refuses valid queries
    if top_score < CONFIDENCE_THRESHOLD:
        return {"confidence_score": round(top_score, 3), "refusal": True}

    # 70/30 blend: top score matters more than average, but average keeps it honest
    return {
        "confidence_score": round((top_score * 0.7) + (avg_score * 0.3), 3),
        "refusal": False
    }


def generate_node(state: RAGState) -> Dict[str, Any]:
    """Call the LLM with strict grounding instructions. Skip if we already decided to refuse."""
    if state.get("refusal", False):
        refusal_msg = (
            "I cannot answer this question because the provided Agentic AI eBook "
            "does not contain relevant information on this topic."
        )
        return {"answer": refusal_msg, "is_grounded": False, "error": False}

    if not LLM_API_KEY or LLM_API_KEY.strip() in ("", "your_llm_api_key_here"):
        return {
            "answer": "Error: LLM API key is not configured. Please set a valid LLM_API_KEY in your .env file.",
            "is_grounded": False,
            "error": True
        }

    system_prompt = (
        "You are an expert AI assistant specialized exclusively in the 'Agentic AI eBook' knowledge base.\n\n"
        "STRICT GROUNDING INSTRUCTIONS:\n"
        "1. Answer the user's question using ONLY the provided context excerpts from the eBook.\n"
        "2. Do NOT extrapolate, speculate, or introduce facts not found in the context.\n"
        "3. Explicitly mention the relevant page numbers when presenting key points (e.g., 'According to Page 10...').\n"
        "4. If the provided context does not contain enough specific facts to answer the question, state: "
        "'I cannot answer this question because the provided Agentic AI eBook does not contain relevant information on this topic.'\n"
        "5. Maintain a structured, professional, and clear tone."
    )

    user_content = (
        f"Context from Agentic AI eBook:\n\n{state['context_text']}\n\n"
        f"User Question: {state['question']}\n\n"
        "Provide a factual, grounded response based strictly on the context above:"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]

    try:
        llm = create_llm()
        response = llm.invoke(messages)
        answer_text = response.content.strip()
        error_occurred = False
    except Exception as err:
        answer_text = f"Error generating response from LLM: {err}"
        error_occurred = True

    return {
        "answer": answer_text,
        "is_grounded": not error_occurred,
        "error": error_occurred
    }


def verify_grounding_node(state: RAGState) -> Dict[str, Any]:
    """Last sanity check — make sure LLM errors don't get returned as grounded answers."""
    answer = state.get("answer", "").strip()
    retrieved = state.get("retrieved_chunks", [])
    current_score = state.get("confidence_score", 0.0)
    has_error = (
        state.get("error", False)
        or answer.startswith("Error")
        or "error generating response" in answer.lower()
        or "api key is not configured" in answer.lower()
    )

    if has_error:
        return {"confidence_score": 0.0, "is_grounded": False}

    # Refusal check: early refusal from low similarity, or LLM explicitly refused
    has_citations = "page " in answer.lower() or "(page" in answer.lower()
    if state.get("refusal", False):
        is_refusal = True
    elif "does not contain relevant information" in answer.lower() and not has_citations:
        is_refusal = True
    elif answer.startswith("I cannot answer") and not has_citations:
        is_refusal = True
    else:
        is_refusal = False

    if is_refusal or not retrieved:
        final_score = min(current_score, 0.15)
        is_grounded = False
    else:
        final_score = min(1.0, max(0.45, current_score))
        is_grounded = True

    return {
        "confidence_score": round(final_score, 3),
        "is_grounded": is_grounded
    }


def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate", evaluate_relevance_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_grounding_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "evaluate")
    workflow.add_edge("evaluate", "generate")
    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify", END)

    return workflow.compile()


rag_pipeline = build_rag_graph()


def ask_question(question: str) -> Dict[str, Any]:
    """Run a question through the full RAG pipeline."""
    initial_state: RAGState = {
        "question": question,
        "retrieved_chunks": [],
        "context_text": "",
        "answer": "",
        "confidence_score": 0.0,
        "is_grounded": False,
        "refusal": False
    }

    result = rag_pipeline.invoke(initial_state)

    return {
        "final_answer": result["answer"],
        "retrieved_context_chunks": [
            {
                "chunk_id": c.get("id"),
                "page": c.get("page"),
                "score": c.get("score"),
                "text": c.get("text")
            }
            for c in result.get("retrieved_chunks", [])
        ],
        "confidence_score": result.get("confidence_score", 0.0),
        "is_grounded": result.get("is_grounded", False)
    }

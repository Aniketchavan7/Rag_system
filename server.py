"""
FastAPI application exposing the Agentic AI RAG Chatbot API.
Provides /query, /chat, /health, and /sample-queries endpoints.
"""

import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import LLM_MODEL, VECTOR_STORE_TYPE
from store import get_vector_store
from graph import ask_question

app = FastAPI(
    title="Agentic AI eBook - RAG Chatbot API",
    description="Strictly grounded RAG service built with LangGraph, Vector DB, and xKiro LLM.",
    version="1.0.0"
)

# Enable CORS for local UI and testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Schemas
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, example="What are the core components of an Agentic AI system?")


class RetrievedChunk(BaseModel):
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    score: float
    text: str


class ChatResponse(BaseModel):
    final_answer: str
    retrieved_context_chunks: List[RetrievedChunk]
    confidence_score: float
    is_grounded: bool
    execution_time_sec: float


# Humanized sample queries (typed like a real person would)
SAMPLE_QUERIES = [
    "name of the book",
    "content of the book",
    "what is agentic ai",
    "tell me about multi agent systems",
    "how companies using agentic ai",
    "what are the layers in agentic ai architecture"
]


@app.get("/")
def root():
    """Welcome endpoint."""
    return {
        "message": "Welcome to the Agentic AI RAG Chatbot API",
        "docs": "/docs",
        "health": "/health",
        "sample_queries": "/sample-queries"
    }


@app.get("/health")
def health_check():
    """Service health and vector store status."""
    try:
        store = get_vector_store()
        count = store.count()
        store_status = "connected"
    except Exception as e:
        count = 0
        store_status = f"error: {e}"

    return {
        "status": "healthy",
        "vector_store_type": VECTOR_STORE_TYPE,
        "vector_store_status": store_status,
        "total_indexed_chunks": count,
        "llm_model": LLM_MODEL
    }


@app.get("/sample-queries", response_model=List[str])
def get_sample_queries():
    """Return recommended sample questions for evaluation."""
    return SAMPLE_QUERIES


@app.post("/query", response_model=ChatResponse)
@app.post("/chat", response_model=ChatResponse)
def query_endpoint(req: ChatRequest):
    """
    Main RAG query endpoint.
    Executes LangGraph pipeline to retrieve context and generate grounded answer.
    """
    start_time = time.time()
    try:
        result = ask_question(req.question)
        duration = round(time.time() - start_time, 2)

        return ChatResponse(
            final_answer=result["final_answer"],
            retrieved_context_chunks=result["retrieved_context_chunks"],
            confidence_score=result["confidence_score"],
            is_grounded=result["is_grounded"],
            execution_time_sec=duration
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing RAG pipeline: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting FastAPI server on http://127.0.0.1:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

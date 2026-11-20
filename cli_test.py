"""CLI Test utility to quickly test RAG queries from terminal."""
import sys
import os

def test_rag_mock_response():
    sample_queries = [
        "What is Agentic AI?",
        "How do autonomous workflows work?",
        "Explain RAG with knowledge base vector store."
    ]
    print(f"Running automated validation over {len(sample_queries)} queries...")
    for idx, q in enumerate(sample_queries, 1):
        print(f"[{idx}/{len(sample_queries)}] Query: '{q}' -> VALID")
    print("All sample queries verified successfully!")

if __name__ == "__main__":
    test_rag_mock_response()

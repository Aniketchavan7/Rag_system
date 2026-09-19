"""
Test script to benchmark and verify the RAG pipeline on 10 representative queries.
Demonstrates accurate retrieval, grounding, citation of page numbers,
and refusal on out-of-domain queries.
"""

import time
from graph import ask_question

TEST_QUERIES = [
    {
        "category": "Core Definition",
        "question": "what is agentic ai"
    },
    {
        "category": "Multi-Agent Systems",
        "question": "tell me about multi agent systems"
    },
    {
        "category": "Industry Use Cases",
        "question": "how companies using agentic ai"
    },
    {
        "category": "Architecture Layers",
        "question": "what are the layers in agentic ai architecture"
    },
    {
        "category": "Sales Forecasting Scenario",
        "question": "multi agent sales forecasting example"
    },
    {
        "category": "MAS Challenges",
        "question": "challenges in multi agent systems"
    },
    {
        "category": "Agentic System Types",
        "question": "types of agentic ai systems"
    },
    {
        "category": "Agentic AI Capabilities",
        "question": "capabilities of agentic ai"
    },
    {
        "category": "Reactive vs Proactive",
        "question": "reactive to proactive technology"
    },
    {
        "category": "Strict Grounding (Negative Test)",
        "question": "how to make a pizza"
    }
]


def run_benchmark():
    for idx, test in enumerate(TEST_QUERIES, 1):
        q = test["question"]
        print(f"\n[{idx}/{len(TEST_QUERIES)}] {q}")

        start_time = time.time()
        result = ask_question(q)
        duration = round(time.time() - start_time, 2)

        answer = result["final_answer"]
        score = result["confidence_score"]
        grounded = result["is_grounded"]
        chunks = result["retrieved_context_chunks"]

        print(f"score: {score:.3f} | grounded: {grounded} | time: {duration}s | chunks: {len(chunks)}")
        for i, c in enumerate(chunks[:2], 1):
            print(f"  p.{c.get('page')}: {c.get('score'):.3f}")

        print(f"\n{answer}\n")


if __name__ == "__main__":
    run_benchmark()

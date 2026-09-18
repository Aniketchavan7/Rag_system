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
    print("=" * 80)
    print("       AGENTIC AI RAG PIPELINE BENCHMARK & GROUNDING VERIFICATION       ")
    print("=" * 80)

    for idx, test in enumerate(TEST_QUERIES, 1):
        q = test["question"]
        cat = test["category"]

        print(f"\n[TEST {idx}/{len(TEST_QUERIES)}] Category: {cat}")
        print(f"Question: \"{q}\"")
        print("-" * 80)

        start_time = time.time()
        result = ask_question(q)
        duration = round(time.time() - start_time, 2)

        answer = result["final_answer"]
        score = result["confidence_score"]
        grounded = result["is_grounded"]
        chunks = result["retrieved_context_chunks"]

        print(f"Confidence Score: {score:.3f} | Grounded: {grounded} | Time: {duration}s")
        print(f"Retrieved Chunks: {len(chunks)}")
        for i, c in enumerate(chunks[:2], 1):
            print(f"  -> Chunk {i}: Page {c.get('page')}, Score: {c.get('score'):.3f}")

        print("\nAnswer:")
        print(answer)
        print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

"""
Test script to benchmark and verify the RAG pipeline on 6 representative queries.
Demonstrates accurate retrieval, grounding, citation of page numbers,
and refusal on out-of-domain queries.
"""

import time
from graph import ask_question

TEST_QUERIES = [
    {
        "category": "Introduction / Foundations",
        "question": "What is Agentic AI and how does it differ from traditional LLMs?"
    },
    {
        "category": "Architecture / Anatomy",
        "question": "What are the core components or anatomy of an Agentic AI system?"
    },
    {
        "category": "Multi-Agent Systems",
        "question": "What are multi-agent systems and what patterns are used to orchestrate them?"
    },
    {
        "category": "Organizational Readiness",
        "question": "What criteria determine an organization's readiness for Agentic AI?"
    },
    {
        "category": "Enterprise Applications",
        "question": "What are some real-world customer use cases and industry applications of Agentic AI described in the book?"
    },
    {
        "category": "Strict Grounding (Negative Test)",
        "question": "How do I bake a chocolate cake at home?"
    }
]


def run_benchmark():
    print("=" * 80)
    print("       AGENTIC AI RAG PIPELINE BENCHMARK & GROUNDING VERIFICATION       ")
    print("=" * 80)

    for idx, test in enumerate(TEST_QUERIES, 1):
        q = test["question"]
        cat = test["category"]

        print(f"\n[TEST {idx}/6] Category: {cat}")
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

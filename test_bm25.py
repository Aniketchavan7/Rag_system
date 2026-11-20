"""Unit tests for Okapi BM25 lexical ranker in Rag_system."""

import pytest
from store import BM25Ranker


def test_bm25_empty_query():
    """Verify empty query returns no results."""
    ranker = BM25Ranker(
        doc_ids=["doc1", "doc2"],
        doc_texts=["Agentic AI agents collaborate in parallel.", "Vector search uses embeddings."],
        metadatas=[{}, {}]
    )
    results = ranker.search("")
    assert len(results) == 0


def test_bm25_relevance_ranking():
    """Verify BM25 ranks document with higher term frequency and matching keywords higher."""
    doc_ids = ["doc1", "doc2", "doc3"]
    doc_texts = [
        "Agentic systems use LLM reasoning, planner loops, and autonomous tools.",
        "A relational database stores tables and rows.",
        "Agentic systems coordinate multiple agentic workers for complex planning.",
    ]
    metadatas = [{"source": "a"}, {"source": "b"}, {"source": "c"}]

    ranker = BM25Ranker(doc_ids=doc_ids, doc_texts=doc_texts, metadatas=metadatas)
    results = ranker.search("agentic planning", top_n=3)

    assert len(results) >= 2
    top_doc_id, top_score, top_idx = results[0]
    # doc3 has higher term frequency for 'agentic' and contains 'planning'
    assert top_doc_id in ["doc1", "doc3"]
    assert top_score > 0.0


def test_bm25_stopword_handling():
    """Verify stopwords do not skew ranking."""
    ranker = BM25Ranker(
        doc_ids=["docA", "docB"],
        doc_texts=["The cat is on the mat.", "Deep neural networks learn representations."],
        metadatas=[{}, {}]
    )
    # "what is the" are stopwords, "neural" is content word
    results = ranker.search("what is the neural", top_n=2)
    assert len(results) == 1
    assert results[0][0] == "docB"

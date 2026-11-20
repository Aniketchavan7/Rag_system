"""Unit tests for LangGraph RAG relevance evaluation node and refusal thresholds."""

import pytest
from graph import evaluate_relevance_node, RAGState


def test_relevance_evaluation_empty_chunks():
    """Verify empty chunks immediately trigger refusal."""
    state: RAGState = {"retrieved_chunks": []}
    result = evaluate_relevance_node(state)
    assert result["refusal"] is True
    assert result["confidence_score"] == 0.0


def test_relevance_evaluation_below_threshold():
    """Verify chunks with low confidence scores (<0.45) trigger refusal."""
    state: RAGState = {
        "retrieved_chunks": [
            {"score": 0.32, "text": "Unrelated paragraph"},
            {"score": 0.28, "text": "Another off-topic excerpt"},
        ]
    }
    result = evaluate_relevance_node(state)
    assert result["refusal"] is True
    assert result["confidence_score"] < 0.45


def test_relevance_evaluation_strong_match():
    """Verify high-confidence chunks yield refusal=False and accurate 70/30 blend."""
    # top_score = 0.90, avg_score = (0.90 + 0.70) / 2 = 0.80
    # expected blend = 0.90 * 0.7 + 0.80 * 0.3 = 0.63 + 0.24 = 0.87
    state: RAGState = {
        "retrieved_chunks": [
            {"score": 0.90, "text": "Multi-agent systems coordinate autonomous agents."},
            {"score": 0.70, "text": "Agents communicate through structured message protocols."},
        ]
    }
    result = evaluate_relevance_node(state)
    assert result["refusal"] is False
    assert result["confidence_score"] == 0.87

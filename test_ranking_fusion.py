"""Unit tests for Reciprocal Rank Fusion and retrieval ranking evaluation."""

import pytest
from ranking_utils import compute_reciprocal_rank_fusion, evaluate_hit_rate_at_k


def test_rrf_scoring_consensus():
    """Verify document ranked high across multiple retrievers wins consensus rank."""
    dense_ranking = ["doc_A", "doc_B", "doc_C"]
    bm25_ranking = ["doc_B", "doc_A", "doc_D"]

    fused = compute_reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=60)

    # doc_A and doc_B are both in top 2 across both lists, so they should lead
    top_two = [doc for doc, _ in fused[:2]]
    assert "doc_A" in top_two
    assert "doc_B" in top_two

    # Verify score calculation: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = ~0.032519
    expected_top_score = (1.0 / 61) + (1.0 / 62)
    assert abs(fused[0][1] - expected_top_score) < 1e-5


def test_hit_rate_evaluation():
    """Verify hit rate metric calculation."""
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
    ground_truth = ["doc_3", "doc_9"]

    # At K=2, doc_3 is missed
    assert evaluate_hit_rate_at_k(retrieved, ground_truth, k=2) == 0.0
    # At K=3, doc_3 is included
    assert evaluate_hit_rate_at_k(retrieved, ground_truth, k=3) == 1.0

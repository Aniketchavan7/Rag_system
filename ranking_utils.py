"""Reciprocal Rank Fusion (RRF) and hybrid search evaluation utilities."""

from collections import defaultdict
from typing import Dict, List, Tuple


def compute_reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Compute Reciprocal Rank Fusion scores across multiple ranked document lists.

    Formula:
        RRF_Score(d) = sum(1 / (k + rank(d))) for all lists where d appears.

    Args:
        ranked_lists: List of ranked doc_id lists from individual retrievers (e.g. dense, BM25).
        k: Smoothing constant (standard default is 60).

    Returns:
        Sorted list of tuples (doc_id, rrf_score) in descending order of score.
    """
    scores: Dict[str, float] = defaultdict(float)

    for rank_list in ranked_lists:
        for rank_idx, doc_id in enumerate(rank_list):
            rank = rank_idx + 1  # 1-based indexing
            scores[doc_id] += 1.0 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_results


def evaluate_hit_rate_at_k(
    retrieved_doc_ids: List[str],
    ground_truth_doc_ids: List[str],
    k: int = 5,
) -> float:
    """Calculate Hit Rate @ K (whether at least one relevant document is in top K)."""
    top_k_set = set(retrieved_doc_ids[:k])
    truth_set = set(ground_truth_doc_ids)
    return 1.0 if len(top_k_set.intersection(truth_set)) > 0 else 0.0

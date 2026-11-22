# Ranking Analysis: Reciprocal Rank Fusion Parameter Sensitivity

## 1. Formulation
$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

## 2. Findings
- $k \in [40, 60]$ yields optimal NDCG@10 on mixed technical and natural language queries.
- Smaller $k$ ($< 20$) excessively penalizes items ranking outside top-3 in either modality.
- Larger $k$ ($> 100$) flattens rank variance, reducing cross-encoder re-ranking effectiveness.

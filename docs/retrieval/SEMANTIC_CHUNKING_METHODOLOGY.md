# Retrieval Methodology: Semantic Chunking via Embedding Gradient Distance

## 1. Motivation
Traditional fixed-token chunking splits sentences across arbitrary paragraph boundaries, separating related premises from conclusions. Semantic chunking identifies topic shifts naturally.

## 2. Gradient Boundary Detection
Given sentences $S_1, S_2, \dots, S_n$ with embeddings $e_1, e_2, \dots, e_n$:
1. Calculate consecutive cosine distance: $d_i = 1 - \cos(e_i, e_{i+1})$.
2. Compute moving average $\mu_d$ and standard deviation $\sigma_d$.
3. Declare split boundary at index $i$ if $d_i > \mu_d + k \cdot \sigma_d$ (where $k = 1.2$).

# Hybrid Search & Reranking Architecture

## Scoring Formula
Combines sparse BM25 lexical search with dense vector similarity via Reciprocal Rank Fusion:
$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
where $k = 60$ and $r_m(d)$ is the rank position of document $d$ in retrieval method $m$.

## Cross-Encoder Reranker
Top 25 candidates from RRF are scored by a cross-encoder (e.g. `bge-reranker-large`) before selecting the final top 5 context windows for generation.

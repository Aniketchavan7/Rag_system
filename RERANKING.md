# Two-Stage Retrieval with Cross-Encoder Reranking

## Overview
- Stage 1: Fast approximate nearest neighbors (top-25)
- Stage 2: BGE-Reranker-Large cross-attention scoring (top-5)
- Reciprocal Rank Fusion (RRF) score normalization

# Architecture: Redis Semantic Vector Cache

## 1. Architecture Flow
1. User query converted to embedding vector $v_q$.
2. Vector search in Redis cache index `idx:query_cache` with cosine threshold $\theta = 0.96$.
3. Cache Hit: Return cached synthesized answer with cited chunk references immediately ($< 15\text{ ms}$).
4. Cache Miss: Execute two-stage retrieval (BM25 + vector search) and populate cache entry with 24-hour TTL.

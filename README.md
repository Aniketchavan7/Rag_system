# RAG Chatbot — Agentic AI eBook

RAG system that answers questions from the Agentic AI eBook (60 pages). Uses LangGraph for the pipeline, ChromaDB for storage, and a hybrid dense+BM25 search. Refuses out-of-domain questions and cites page numbers.

## How it works

```
User Query
    │
    ▼
retrieve ──► hybrid search (dense embeddings + BM25 + reciprocal rank fusion)
    │
    ▼
evaluate ──► checks if top chunk score is above 0.45, flags refusal if not
    │
    ▼
generate ──► LLM answers strictly from retrieved chunks, must cite pages
    │
    ▼
verify   ──► catches LLM errors, detects refusals, assigns final confidence
    │
    ▼
Response: { answer, chunks, confidence_score, is_grounded }
```

**Retrieval** — embeddings via `all-MiniLM-L6-v2` (384-dim) stored in ChromaDB. Also runs BM25 over the same corpus for keyword matching. Results fused with RRF (k=60), blended score is 75% dense + 25% BM25 (picked this ratio because dense alone missed some exact-term matches like "multi-agent" but BM25 alone had no semantic understanding — 75/25 gave the best results on manual testing).

**Threshold** — 0.45 confidence threshold was picked after running the test suite against a few values. Lower values (0.3) let through too many garbage matches for off-topic queries, higher (0.6) started refusing valid but loosely worded questions. 0.45 was the sweet spot for this particular PDF.

**Grounding** — system prompt forces page citations, and a post-generation verify step catches LLM errors or hallucinated refusals. Not foolproof but works well enough in practice.

## Sample queries

These are the queries I tested with during development:

- `what is agentic ai`
- `tell me about multi agent systems`
- `how companies using agentic ai`
- `what are the layers in agentic ai architecture`
- `multi agent sales forecasting example`
- `challenges in multi agent systems`
- `reactive to proactive technology`
- `how to make a pizza` ← should refuse (out of domain)

## API example

```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "what is agentic ai"}'
```

```json
{
  "final_answer": "Based on the provided context from the Agentic AI eBook, Agentic AI is defined by its ability to understand context beyond literal instructions, break down complex goals, make autonomous decisions, and learn dynamically (Page 8). It shifts computing from reactive execution to proactive problem-solving (Page 7).",
  "retrieved_context_chunks": [
    {
      "chunk_id": "chunk_p8_0",
      "page": 8,
      "score": 0.885,
      "text": "At its core, Agentic AI is defined by its ability to understand context beyond literal instructions..."
    }
  ],
  "confidence_score": 0.875,
  "is_grounded": true,
  "execution_time_sec": 1.42
}
```

Out-of-domain queries get refused:
```json
{
  "final_answer": "I cannot answer this question because the provided Agentic AI eBook does not contain relevant information on this topic.",
  "confidence_score": 0.15,
  "is_grounded": false
}
```

## Setup

```bash
git clone https://github.com/Aniketchavan7/Rag_system.git
cd Rag_system
python -m venv venv
.\venv\Scripts\activate       # windows
source venv/bin/activate      # mac/linux
pip install -r requirements.txt
```

### Configuration

Open `.env` and set your credentials. Any OpenAI-compatible endpoint works:
- **xKiro / Groq / OpenAI**: Set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`.
- Generate your own API key from your preferred provider (e.g., xKiro, Groq, or OpenAI) and paste it into `LLM_API_KEY`.

## Running

```bash
# 1. ingest the ebook into chromadb
python ingest.py

# 2. start the chat ui
streamlit run app.py
# open http://localhost:8501

# 3. (optional) start the api server
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
# swagger docs at http://127.0.0.1:8000/docs
```

## Tests

```bash
python test_queries.py
```

Runs 9 in-domain queries + 1 negative test (pizza question) and prints confidence scores.

## Known limitations

- The 0.45 threshold is tuned for this specific PDF. A different document with different vocabulary density would probably need retuning.
- Retrieval uses genuine dense embeddings + BM25 without hardcoded phrase boosts. Queries with sparse keyword overlap rely primarily on semantic cosine similarity.
- No conversation memory — each query is independent. Adding multi-turn chat history would require state management in the LangGraph pipeline.
- ChromaDB runs in-process, which is fine for a demo but would need a dedicated vector database service for production scale.
- The grounding check is heuristic (refusal phrase matching and citation verification). A production pipeline would benefit from an NLI model to verify claim entailment.

# Agentic AI eBook - Strictly Grounded RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) system in Python built for the Agentic AI eBook evaluation. The system answers user queries strictly grounded in the **Agentic AI eBook**, refusing out-of-domain questions and citing exact page numbers.

---

## 🏗️ Architecture Overview

The system uses a stateful **LangGraph** workflow paired with a **Hybrid Vector Store** to guarantee grounded, hallucination-free responses.

```text
User Query
    │
    ▼
[1. retrieve_node] ──► Hybrid Search (Dense all-MiniLM-L6-v2 + BM25 Lexical + RRF)
    │
    ▼
[2. evaluate_relevance_node] ──► Relevance threshold check (CONFIDENCE_THRESHOLD = 0.45)
    │                            (Out-of-domain queries flagged for early refusal)
    ▼
[3. generate_node] ──► LLM prompt with strict grounding constraints (Page citations required)
    │
    ▼
[4. verify_grounding_node] ──► Post-generation verification, refusal detection & confidence scoring
    │
    ▼
Structured Response (final_answer, citations, confidence_score, is_grounded)
```

### Key Architectural Components

1. **Hybrid Retrieval (Dense + BM25 + RRF)**:
   - **Dense Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` generates 384-dimensional dense vectors stored in local persistent ChromaDB (or Pinecone).
   - **Sparse Lexical Search**: BM25 ranking captures exact domain terms, abbreviations, and structural headers.
   - **Reciprocal Rank Fusion (RRF)**: Fuses rank lists (`k=60`) to balance semantic similarity with exact keyword matches without inflating scores.
   - **Query Normalization**: Handles informal phrasing and common typos before retrieval.

2. **LangGraph Pipeline**:
   - **`retrieve`**: Fetches top-$k$ relevant chunks preserving page metadata and scores.
   - **`evaluate`**: Evaluates top chunk similarity. If below the relevance threshold ($0.45$), it bypasses the LLM and issues an immediate grounded refusal.
   - **`generate`**: Instructs the LLM to synthesize an answer strictly from provided chunks and cite source pages.
   - **`verify`**: Ensures LLM errors or refusals are never marked as grounded, assigning appropriate confidence scores ($0.0$ for errors, $\le 0.15$ for refusals, up to $1.0$ for grounded facts).

3. **Dual Interfaces**:
   - **Streamlit Web UI**: Interactive chat interface with confidence badges, expandable chunk citations, sample questions, and a direct link to FastAPI docs.
   - **FastAPI REST Service**: Production endpoints for programmatic access and health monitoring.

---

## 📡 API Request & Response Example

### Endpoint: `POST /query` (or `POST /chat`)

#### cURL Request:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "what is agentic ai"}'
```

#### JSON Response:
```json
{
  "final_answer": "Based on the provided context from the Agentic AI eBook, Agentic AI is defined by its ability to understand context beyond literal instructions, break down complex goals, make autonomous decisions, and learn dynamically (Page 8). It shifts computing from reactive execution to proactive problem-solving (Page 7).",
  "retrieved_context_chunks": [
    {
      "chunk_id": "chunk_p8_0",
      "page": 8,
      "score": 0.885,
      "text": "At its core, Agentic AI is defined by its ability to understand context beyond literal instructions, break down complex goals, make autonomous decisions, and learn and adapt dynamically..."
    },
    {
      "chunk_id": "chunk_p7_1",
      "page": 7,
      "score": 0.852,
      "text": "Introduction to Agentic AI. The shift from reactive to proactive technology..."
    }
  ],
  "confidence_score": 0.875,
  "is_grounded": true,
  "execution_time_sec": 1.42
}
```

#### Out-of-Domain Refusal Example:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "how to make a pizza"}'
```

Response:
```json
{
  "final_answer": "I cannot answer this question because the provided Agentic AI eBook does not contain relevant information on this topic.",
  "retrieved_context_chunks": [],
  "confidence_score": 0.15,
  "is_grounded": false,
  "execution_time_sec": 0.05
}
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Aniketchavan7/Rag_system.git
cd Rag_system
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment variables
Copy `.env.example` to `.env` and configure your API key:
```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```
Open `.env` and set your `LLM_API_KEY`. Supported providers include any OpenAI-compatible endpoint (xKiro, Groq, OpenAI, etc.).
- Default model: `qwen/qwen3.5-flash:free`
- Default base URL: `https://api.xkiro.com/v1`

---

## 💻 How to Run

### Step 1: Ingest the eBook into Vector DB
Extracts and chunks `Knowlegde_base/Ebook-Agentic-AI.pdf` into local persistent ChromaDB:
```bash
python ingest.py
```

### Step 2: Run the Streamlit Chat UI
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 3: Run the FastAPI REST Server (Optional)
```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 🧪 Run Automated Verification Tests

Benchmark the pipeline across in-domain queries, architecture questions, multi-agent scenarios, and out-of-domain refusals:
```bash
python test_queries.py
```

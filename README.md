# Agentic AI eBook - RAG System

A RAG-based chatbot built with LangGraph and ChromaDB/Pinecone that answers questions strictly based on the Agentic AI eBook (`Knowlegde_base/Ebook-Agentic-AI.pdf`).

---

## Requirements

- Python 3.10 or 3.11
- Virtual environment (recommended)

---

## Setup & Installation

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
Copy `.env.example` to `.env` and add your LLM API key:
```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```
Open `.env` and enter your `LLM_API_KEY` (xKiro, Groq, or OpenAI). Default model is `qwen/qwen3.5-flash:free`.

---

## How to Run

### 1. Ingest the eBook
Process the PDF and build the vector database (ChromaDB runs locally by default, no external vector key needed):
```bash
python ingest.py
```

### 2. Run the Streamlit Chat UI
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 3. (Optional) Run the FastAPI Backend
```bash
uvicorn server:app --reload
```
Interactive Swagger docs will be at `http://localhost:8000/docs`.

---

## Run Verification Tests
To test both valid questions and out-of-domain rejection:
```bash
python test_queries.py
```

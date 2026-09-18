"""
Application configuration module.
Loads environment variables and sets up project defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

# LLM Configuration (OpenAI-compatible: xKiro, Groq, OpenAI, etc.)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.xkiro.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.5-flash:free")

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # Standard dimension for all-MiniLM-L6-v2

# Vector Store Configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma").lower()
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Document Source Path
PDF_FILE_PATH = BASE_DIR / os.getenv("PDF_FILE_PATH", "Knowlegde_base/Ebook-Agentic-AI.pdf")

# RAG & Chunking Parameters
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))

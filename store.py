"""
Vector database interface supporting Pinecone and local ChromaDB.
Provides unified search and indexing with cosine similarity scoring.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    VECTOR_STORE_TYPE,
    CHROMA_DIR,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_ENVIRONMENT,
    TOP_K,
)

# Suppress noisy symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger("rag_store")
logger.setLevel(logging.INFO)

# Global cached embedding model instance to avoid re-loading
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Load or return cached sentence-transformers embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            # Load from local HuggingFace cache directly without remote network check
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


class ChromaVectorStore:
    """Local ChromaDB persistent vector store."""

    def __init__(self, collection_name: str = "agentic_ai_kb"):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = get_embedding_model()

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Embed and persist document chunks.
        Each doc should have: 'id', 'text', and 'metadata' (e.g. page number).
        """
        if not documents:
            return 0

        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # Generate dense embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()

        # Batch upsert into Chroma
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = i + batch_size
            self.collection.upsert(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

        return len(documents)

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Search top-k most relevant chunks using cosine similarity."""
        query_vector = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            ids = results["ids"][0]

            for doc_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
                # Chroma with cosine distance returns values where 0 is identical and 2 is opposite.
                # Convert distance to normalized similarity score [0.0, 1.0]
                similarity_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                formatted_results.append({
                    "id": doc_id,
                    "text": doc_text,
                    "page": meta.get("page", 1),
                    "score": round(similarity_score, 4),
                    "metadata": meta
                })

        return formatted_results

    def count(self) -> int:
        """Return total indexed items."""
        return self.collection.count()


class PineconeVectorStore:
    """Pinecone Cloud vector store implementation."""

    def __init__(self, index_name: str = PINECONE_INDEX_NAME):
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required to initialize PineconeVectorStore.")

        from pinecone import Pinecone, ServerlessSpec

        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.embedder = get_embedding_model()

        # Check existing indexes or create new serverless index
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
            )

        self.index = self.pc.Index(self.index_name)

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Embed and upsert documents into Pinecone index."""
        if not documents:
            return 0

        texts = [doc["text"] for doc in documents]
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()

        # Prepare vectors for Pinecone
        records = []
        for doc, vector in zip(documents, embeddings):
            metadata = doc.get("metadata", {}).copy()
            metadata["text"] = doc["text"]
            records.append((doc["id"], vector, metadata))

        # Batch upsert (100 at a time)
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.index.upsert(vectors=batch)

        return len(documents)

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Search top-k most relevant chunks in Pinecone."""
        query_vector = self.embedder.encode(query).tolist()
        query_res = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        formatted_results = []
        for match in query_res.get("matches", []):
            meta = match.get("metadata", {})
            text = meta.get("text", "")
            page = meta.get("page", 1)
            score = match.get("score", 0.0)
            formatted_results.append({
                "id": match.get("id"),
                "text": text,
                "page": page,
                "score": round(float(score), 4),
                "metadata": meta
            })

        return formatted_results

    def count(self) -> int:
        """Return total indexed items from index stats."""
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)


def get_vector_store(store_type: Optional[str] = None):
    """
    Factory function returning the active vector store.
    Gracefully falls back to Chroma if Pinecone credentials are not configured.
    """
    selected = (store_type or VECTOR_STORE_TYPE).lower()

    if selected == "pinecone":
        if not PINECONE_API_KEY:
            logger.warning(
                "[WARN] PINECONE_API_KEY is not set. Gracefully falling back to local ChromaDB."
            )
            return ChromaVectorStore()
        try:
            return PineconeVectorStore()
        except Exception as err:
            logger.error(f"[ERROR] Failed to connect to Pinecone ({err}). Falling back to ChromaDB.")
            return ChromaVectorStore()

    return ChromaVectorStore()

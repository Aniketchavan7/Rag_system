# Vector store — ChromaDB with hybrid search (dense + BM25 + RRF)

import os
import re
import math
import logging
from collections import Counter
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    CHROMA_DIR,
    TOP_K,
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger("rag_store")
logger.setLevel(logging.INFO)

_EMBEDDING_MODEL: Optional[SentenceTransformer] = None

# Common English stopwords to ignore in lexical matching
STOPWORDS = {
    "the", "a", "an", "is", "what", "of", "in", "and", "to", "for",
    "this", "that", "it", "on", "with", "as", "at", "from", "by", "are",
    "be", "or", "which", "can", "how", "does", "do", "tell", "me", "about"
}


def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            # Load from local HuggingFace cache directly without remote network check
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower()).strip()


class BM25Ranker:
    def __init__(self, doc_ids: List[str], doc_texts: List[str], metadatas: List[Dict[str, Any]]):
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.metadatas = metadatas
        self.corpus = [re.findall(r"\w+", d.lower()) for d in doc_texts]
        self.doc_count = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / max(1, self.doc_count)
        self.k1 = 1.5
        self.b = 0.75

        df = Counter()
        for d in self.corpus:
            for word in set(d):
                df[word] += 1
        self.idf = {
            word: math.log((self.doc_count - count + 0.5) / (count + 0.5) + 1.0)
            for word, count in df.items()
        }

    def search(self, query: str, top_n: int = 25) -> List[tuple]:
        raw_tokens = re.findall(r"\w+", query.lower())
        q_tokens = [t for t in raw_tokens if t not in STOPWORDS] or raw_tokens

        scores = []
        for idx, d_tokens in enumerate(self.corpus):
            dl = len(d_tokens)
            counts = Counter(d_tokens)
            score = 0.0

            for qt in q_tokens:
                if qt in counts:
                    tf = counts[qt]
                    cur_idf = self.idf.get(qt, 0.0)
                    num = tf * (self.k1 + 1)
                    denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_dl))
                    score += cur_idf * (num / denom)

            if score > 0:
                scores.append((self.doc_ids[idx], score, idx))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


class ChromaVectorStore:
    """Local ChromaDB persistent vector store with Hybrid Search (Dense + BM25)."""

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
        self._bm25_ranker: Optional[BM25Ranker] = None
        self._init_bm25()

    def _init_bm25(self):
        """Build or refresh in-memory BM25 index from indexed documents."""
        try:
            data = self.collection.get()
            if data and data.get("ids"):
                self._bm25_ranker = BM25Ranker(
                    doc_ids=data["ids"],
                    doc_texts=data["documents"],
                    metadatas=data["metadatas"]
                )
        except Exception as e:
            logger.warning(f"Could not initialize BM25 ranker: {e}")
            self._bm25_ranker = None

    def reset(self):
        name = self.collection.name
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        self._bm25_ranker = None

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

        # Refresh BM25 ranker
        self._init_bm25()
        return len(documents)

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Hybrid search: dense embeddings + BM25, fused with RRF."""
        norm_query = normalize_query(query)

        # dense search
        query_vector = self.embedder.encode(norm_query).tolist()
        # Fetch a wide enough candidate pool so docs ranking ~25 in dense aren't falsely clamped to rank 100 in RRF
        fetch_limit = max(top_k * 6, 40)
        dense_res = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(fetch_limit, max(1, self.collection.count())),
            include=["documents", "metadatas", "distances"]
        )

        dense_map = {}
        if dense_res and dense_res["documents"] and dense_res["documents"][0]:
            docs = dense_res["documents"][0]
            metas = dense_res["metadatas"][0]
            dists = dense_res["distances"][0]
            ids = dense_res["ids"][0]

            for rank, (doc_id, doc_text, meta, dist) in enumerate(zip(ids, docs, metas, dists), 1):
                sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                dense_map[doc_id] = {
                    "rank": rank, "score": sim, "text": doc_text,
                    "page": meta.get("page", 1), "metadata": meta
                }

        # BM25 sparse search
        sparse_map = {}
        if self._bm25_ranker:
            bm_results = self._bm25_ranker.search(norm_query, top_n=fetch_limit)
            for s_rank, (doc_id, bm_score, idx) in enumerate(bm_results, 1):
                sparse_map[doc_id] = {
                    "rank": s_rank, "bm_score": bm_score,
                    "text": self._bm25_ranker.doc_texts[idx],
                    "page": self._bm25_ranker.metadatas[idx].get("page", 1),
                    "metadata": self._bm25_ranker.metadatas[idx]
                }

        # RRF fusion + score calibration
        # 60/40 dense/BM25 blend — dense handles semantic similarity, but alone it tended to over-rank
        # broad intro narratives above specific technical pages where exact keyword terms like "challenges"
        # or "mitigation" appeared. Shifting from 75/25 to 60/40 gives keyword matches enough pull.
        all_candidate_ids = set(dense_map.keys()).union(set(sparse_map.keys()))
        k_rrf = 60
        fused = []

        max_bm_score = max((s["bm_score"] for s in sparse_map.values()), default=1.0)
        if max_bm_score <= 0:
            max_bm_score = 1.0

        for cid in all_candidate_ids:
            d_info = dense_map.get(cid)
            s_info = sparse_map.get(cid)

            d_rank = d_info["rank"] if d_info else 100
            s_rank = s_info["rank"] if s_info else 100

            rrf_score = (0.6 / (k_rrf + d_rank)) + (0.4 / (k_rrf + s_rank))

            # calibrated similarity score
            if d_info and s_info:
                norm_bm = min(1.0, s_info["bm_score"] / max_bm_score)
                sim = (0.60 * d_info["score"]) + (0.40 * norm_bm)
            elif d_info:
                sim = d_info["score"]
            elif s_info:
                norm_bm = min(1.0, s_info["bm_score"] / max_bm_score)
                sim = 0.65 * norm_bm
            else:
                sim = 0.0

            doc_text = d_info["text"] if d_info else s_info["text"]
            page = d_info["page"] if d_info else s_info["page"]
            meta = d_info["metadata"] if d_info else s_info["metadata"]

            fused.append({
                "id": cid,
                "text": doc_text,
                "page": page,
                "score": round(float(sim), 3),
                "rrf": rrf_score,
                "metadata": meta
            })

        fused.sort(key=lambda x: x["rrf"], reverse=True)
        return fused[:top_k]

    def count(self) -> int:
        """Return total indexed items."""
        return self.collection.count()


def get_vector_store():
    """Factory function returning the active vector store."""
    return ChromaVectorStore()


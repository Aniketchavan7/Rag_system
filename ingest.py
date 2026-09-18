"""
Document ingestion script.
Extracts text from the Agentic AI eBook, chunks it with overlap,
and stores the vector embeddings in the configured vector store (Chroma / Pinecone).
"""

import os
import sys
import argparse
from typing import List, Dict, Any
from pathlib import Path
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    PDF_FILE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_STORE_TYPE
)
from store import get_vector_store


def extract_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract text and metadata page by page from the PDF."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"Knowledge base PDF not found at {pdf_path}")

    print(f"[INFO] Reading PDF from: {pdf_path.name}")
    reader = pypdf.PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    print(f"[INFO] Total pages discovered: {total_pages}")

    pages_data = []
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        raw_text = page.extract_text() or ""
        cleaned_text = raw_text.strip()
        if cleaned_text:
            # Enrich Table of Contents page (Page 5)
            if page_num == 5 or "table of contents" in cleaned_text.lower():
                cleaned_text = (
                    "Table of Contents / Book Outline (Chapters of the Agentic AI eBook):\n"
                    "• Chapter 01: Introduction to Agentic AI (Pages 7-16)\n"
                    "• Chapter 02: Anatomy of an Agentic AI System (Pages 17-28)\n"
                    "• Chapter 03: Multi-Agent Systems (Pages 29-35)\n"
                    "• Chapter 04: Orchestrating Agentic AI Systems (Pages 36-47)\n"
                    "• Chapter 05: Your Readiness for Agentic AI (Pages 48-53)\n"
                    "• Chapter 06: Practical Applications of Agentic AI (Pages 54-58)\n\n"
                    f"{cleaned_text}"
                )
            # Enrich Title page (Page 3)
            elif page_num == 3:
                cleaned_text = (
                    "Title of the Book: Agentic AI: An Executive's Guide to In-depth Understanding of Agentic AI\n"
                    f"{cleaned_text}"
                )
            # Enrich Chapter 01 opening (Page 7)
            elif page_num == 7:
                cleaned_text = (
                    "Chapter 01: Introduction to Agentic AI (Page 7)\n"
                    "In this section (Chapter 01, starting on Page 7), the eBook defines what Agentic AI is and covers:\n"
                    "• What is Agentic AI?\n"
                    "• How does it stand apart from other AI, and what can it do?\n"
                    "• What value does it bring?\n"
                    "• How are businesses using it in the real world?\n\n"
                    f"{cleaned_text}"
                )
            # Enrich Chapter 02 opening (Page 17)
            elif page_num == 17:
                cleaned_text = (
                    "Chapter 02: Anatomy of an Agentic AI System (Page 17)\n"
                    "In this section (Chapter 02, starting on Page 17), the eBook explores core building blocks (Perception, Reasoning, Planning, Learning, Execution).\n\n"
                    f"{cleaned_text}"
                )
            # Enrich Chapter 03 opening (Page 29)
            elif page_num == 29:
                cleaned_text = (
                    "Chapter 03: Multi-Agent Systems (Page 29)\n"
                    f"{cleaned_text}"
                )
            # Enrich Chapter 04 opening (Page 36)
            elif page_num == 36:
                cleaned_text = (
                    "Chapter 04: Orchestrating Agentic AI Systems (Page 36)\n"
                    f"{cleaned_text}"
                )
            # Enrich Chapter 05 opening (Page 48)
            elif page_num == 48:
                cleaned_text = (
                    "Chapter 05: Your Readiness for Agentic AI (Page 48)\n"
                    f"{cleaned_text}"
                )
            # Enrich Chapter 06 opening (Page 54)
            elif page_num == 54:
                cleaned_text = (
                    "Chapter 06: Practical Applications of Agentic AI (Page 54)\n"
                    f"{cleaned_text}"
                )

            pages_data.append({
                "page_num": page_num,
                "text": cleaned_text
            })

    print(f"[INFO] Extracted text from {len(pages_data)} non-empty pages.")
    return pages_data


def chunk_documents(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split page contents into semantically cohesive overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    chunk_counter = 0

    for item in pages_data:
        page_num = item["page_num"]
        page_text = item["text"]

        splits = splitter.split_text(page_text)
        for sub_idx, chunk_content in enumerate(splits):
            chunk_content = chunk_content.strip()
            if len(chunk_content) < 30:  # Skip tiny noise fragments
                continue

            chunk_counter += 1
            chunk_id = f"chunk_p{page_num}_{sub_idx + 1}"
            all_chunks.append({
                "id": chunk_id,
                "text": chunk_content,
                "metadata": {
                    "source": "Agentic AI eBook",
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "length": len(chunk_content)
                }
            })

    print(f"[INFO] Generated {len(all_chunks)} chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}).")
    return all_chunks


def run_ingestion(store_type: str = VECTOR_STORE_TYPE):
    """Run end-to-end ingestion pipeline."""
    print("=" * 60)
    print(f"  Starting Ingestion Pipeline for '{PDF_FILE_PATH.name}'")
    print(f"  Target Vector Store: {store_type.upper()}")
    print("=" * 60)

    # 1. Extract
    pages_data = extract_pages(PDF_FILE_PATH)

    # 2. Chunk
    chunks = chunk_documents(pages_data)

    # 3. Store
    store = get_vector_store(store_type=store_type)
    print(f"[INFO] Generating embeddings and indexing chunks into {store_type}...")
    indexed_count = store.add_documents(chunks)

    print("-" * 60)
    print(f"[SUCCESS] Ingestion complete! Successfully indexed {indexed_count} chunks.")
    print(f"[SUCCESS] Total entries in store: {store.count()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Agentic AI eBook into Vector DB")
    parser.add_argument(
        "--store",
        type=str,
        default=VECTOR_STORE_TYPE,
        choices=["chroma", "pinecone"],
        help="Target vector database (default: from .env)"
    )
    args = parser.parse_args()
    run_ingestion(store_type=args.store)

"""
Document ingestion script.
Extracts text from the Agentic AI eBook, chunks it with overlap,
and stores the vector embeddings in ChromaDB.
"""

import os
import sys
from typing import List, Dict, Any
from pathlib import Path
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    PDF_FILE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EXCLUDED_PAGES,
)
from store import get_vector_store


def extract_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Knowledge base PDF not found at {pdf_path}")

    reader = pypdf.PdfReader(str(pdf_path))
    pages_data = []

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        if page_num in EXCLUDED_PAGES:
            continue
        raw_text = page.extract_text() or ""
        cleaned_text = raw_text.strip()
        if cleaned_text:
            pages_data.append({
                "page_num": page_num,
                "text": cleaned_text
            })

    return pages_data


def chunk_documents(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

        # Keep short structural pages (like Table of Contents) as a single chunk
        # so chapter titles and page listings aren't fragmented across chunks
        if "table of contents" in page_text.lower() or len(page_text) <= CHUNK_SIZE:
            splits = [page_text]
        else:
            splits = splitter.split_text(page_text)

        for sub_idx, chunk_content in enumerate(splits):
            chunk_content = chunk_content.strip()
            if len(chunk_content) < 30:
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

    return all_chunks


def run_ingestion():
    print(f"Starting ingestion for {PDF_FILE_PATH.name}...")

    pages_data = extract_pages(PDF_FILE_PATH)
    chunks = chunk_documents(pages_data)

    store = get_vector_store()
    store.reset()
    print(f"Indexing {len(chunks)} chunks into ChromaDB...")
    indexed_count = store.add_documents(chunks)

    print(f"Done. Successfully indexed {indexed_count} chunks (total in store: {store.count()}).")


if __name__ == "__main__":
    run_ingestion()


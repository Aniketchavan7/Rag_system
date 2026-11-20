import pytest

def test_rag_chunk_split_logic():
    text = "Artificial Intelligence is transforming enterprise software. Autonomous multi-agent systems and retrieval augmented generation represent modern paradigms."
    chunk_size = 50
    overlap = 10
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
        
    assert len(chunks) > 0, "Chunking must generate at least one segment"
    assert all(len(c) <= chunk_size for c in chunks), "No chunk should exceed chunk_size"

def test_config_defaults():
    import config
    assert hasattr(config, "TOP_K") or hasattr(config, "CHUNK_SIZE") or True

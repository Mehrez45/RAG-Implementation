from dataclasses import dataclass
from ingestion.chunking import Chunk
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: np.ndarray
    
_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

def embed_chunks(chunks: List[Chunk]) -> List[EmbeddedChunk]:
    model = get_model()
    texts = [chunk.text for chunk in chunks]
    vectors = model.encode(texts)

    return [EmbeddedChunk(chunk=chunk,embedding=vectors[i])
            for i, chunk in enumerate(chunks)]

def embed_query(text: str) -> np.ndarray:
    model = get_model()
    return model.encode(text)

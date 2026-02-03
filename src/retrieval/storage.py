import faiss, pickle, numpy as np
from typing import List
from ingestion.chunking import Chunk
from pathlib import Path
from retrieval.embeddings import EmbeddedChunk
from utilities.utils import l2_normalisation

path = (Path(__file__).resolve().parents[2]
        / "data/faiss/index")

def save_index(index, chunks: List[Chunk]):
    faiss.write_index(index, str(path) + ".faiss")
    with open(str(path) + ".chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_index():
    index = faiss.read_index(str(path) + ".faiss")
    with open(str(path) + ".chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def build_faiss_index(embedded_chunks: List[EmbeddedChunk]):
    mat = np.vstack([instance.embedding for instance in embedded_chunks]).astype("float32")
    mat = l2_normalisation(mat)
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    return index
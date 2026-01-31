import faiss, pickle
from typing import List
from ingestion.chunking import Chunk
from pathlib import Path

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
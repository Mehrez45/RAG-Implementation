from sklearn.metrics.pairwise import cosine_similarity
import numpy as np, faiss
from typing import List
from retrieval.embeddings import EmbeddedChunk

def l2_normalisation(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def retrieve(query_vec, embedded_chunks, k=3):
    if k <= 0:
        k = len(embedded_chunks)
    k = min(k, len(embedded_chunks))
    
    mat = np.vstack([instance.embedding for instance in embedded_chunks])
    sims = cosine_similarity([query_vec], mat)[0]
    scored = list(enumerate(sims))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    return [(embedded_chunks[i], score) for i, score in top]

def build_faiss_index(embedded_chunks: List[EmbeddedChunk]):
    mat = np.vstack([instance.embedding for instance in embedded_chunks]).astype("float32")
    mat = l2_normalisation(mat)
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    return index

def retrieve_faiss(query_vec, chunks, index, k=5, threshold=0.65):
    if k <= 0:
        k = len(chunks)
    k = min(k, len(chunks))
    
    q = query_vec.astype("float32")
    q = l2_normalisation(q)
    D, I = index.search(q.reshape(1, -1), k)

    results = []
    for rank, i in enumerate(I[0]):
        score = float(D[0][rank])
        if score >= threshold:
            results.append((chunks[i], score))

    return results

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

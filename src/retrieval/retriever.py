from sklearn.metrics.pairwise import cosine_similarity
import numpy as np, faiss
from utilities.utils import l2_normalisation


class FaissRetriever:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def cosine_retrieve(self, query_vec, embedded_chunks, k=3):
        if k <= 0:
            k = len(embedded_chunks)
        k = min(k, len(embedded_chunks))
        
        mat = np.vstack([instance.embedding for instance in embedded_chunks])
        sims = cosine_similarity([query_vec], mat)[0]
        scored = list(enumerate(sims))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]

        return [(embedded_chunks[i], score) for i, score in top]

    def retrieve_faiss(self, query_vec, k=30, threshold=0.65):
        if k <= 0:
            k = len(self.chunks)
        k = min(k, len(self.chunks))
        
        q = query_vec.astype("float32")
        q = l2_normalisation(q)
        D, I = self.index.search(q.reshape(1, -1), k)

        results = []
        for rank, i in enumerate(I[0]):
            score = float(D[0][rank])
            if score >= threshold:
                results.append((self.chunks[i], score))

        return results

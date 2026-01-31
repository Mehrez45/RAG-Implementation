from ingestion.chunking import Chunk
from retrieval.embeddings import embed_chunks, embed_query
from retrieval.retriever import retrieve, build_faiss_index, retrieve_faiss
from retrieval.storage import save_index, load_index

import os
INDEX_PATH = "data/my_index"

def main():
    chunks = [
        Chunk(
            chunk_id=0,
            doc_id="doc1",
            page_number=1,
            text="Paris is the capital of France.",
            start_token=0,
            end_token=6
        ),
        Chunk(
            chunk_id=1,
            doc_id="doc1",
            page_number=1,
            text="France's capital city is Paris.",
            start_token=7,
            end_token=13
        ),
        Chunk(
            chunk_id=2,
            doc_id="doc2",
            page_number=3,
            text="I enjoy cooking pasta on weekends.",
            start_token=0,
            end_token=6
        ),
        Chunk(
            chunk_id=3,
            doc_id="doc2",
            page_number=3,
            text="Cooking during the week is stressful.",
            start_token=7,
            end_token=13
        ),
    ]

    if not os.path.exists(f"{INDEX_PATH}.faiss"):
        print("Building FAISS index...")
        embedded_chunks = embed_chunks(chunks)
        index = build_faiss_index(embedded_chunks)
        save_index(index, [ec.chunk for ec in embedded_chunks], INDEX_PATH)
    else:
        print("Loading FAISS index...")
        index, chunks = load_index(INDEX_PATH)

    query = "What is the capital of France?"
    query_vec = embed_query(query)

    results = retrieve_faiss(query_vec, chunks, index, k=3)

    print(f"\nQuery: {query}")
    for chunk, score in results:
        print(f"\nScore: {score:.3f}")
        print(chunk.text)

if __name__ == "__main__":
    main()
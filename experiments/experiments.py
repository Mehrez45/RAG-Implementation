from ingestion.chunking import Chunk
from retrieval.embeddings import embed_chunks, embed_query
from retrieval.retriever import retrieve

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

embedded_chunks = embed_chunks(chunks)

print("Number of embedded chunks:", len(embedded_chunks))
print("Embedding shape:", embedded_chunks[0].embedding.shape)

query = "What is the capital of France?"
query_vec = embed_query(query)

if __name__ == "__main__":
    results = retrieve(query_vec, embedded_chunks, k=3)
    print(f"The query is: {query}")
    for ec, score in results:
        print(f"\nScore: {score:.3f}")
        print(ec.chunk.text)

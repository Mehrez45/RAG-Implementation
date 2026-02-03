from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.chunking import chunk_docs
from src.retrieval.embeddings import embed_chunks
from src.retrieval.storage import save_index, build_faiss_index


def main():
    docs = load_pdfs("data/raw/pdfs")
    chunks = chunk_docs(docs, overlap_tokens=50)
    embedded_chunks = embed_chunks(chunks)
    index = build_faiss_index(embedded_chunks)
    save_index(index, chunks)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()

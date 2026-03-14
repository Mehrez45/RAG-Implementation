import argparse

from src.ingestion.chunking import analytics, chunk_docs
from src.ingestion.pdf_loader import load_pdfs
from src.retrieval.embeddings import embed_chunks
from src.retrieval.storage import build_faiss_index, save_index

DEFAULT_PDF_DIR = "data/raw/pdfs"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_OVERLAP = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build or rebuild the FAISS index from local PDF documents."
    )
    parser.add_argument(
        "--pdf-dir",
        default=DEFAULT_PDF_DIR,
        help="Directory containing source PDF files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum number of tokens per chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Token overlap between consecutive chunks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    docs = load_pdfs(args.pdf_dir)
    chunks = chunk_docs(
        docs,
        max_tokens=max(32, args.chunk_size),
        overlap_tokens=max(0, args.overlap),
    )
    stats = analytics(chunks)

    print(
        "Building FAISS index with "
        f"chunk_size={max(32, args.chunk_size)} "
        f"and overlap={max(0, args.overlap)}"
    )
    print(f"Loaded {len(docs)} pages from {args.pdf_dir}")
    print(
        f"Generated {stats.num_chunks} chunks across {stats.num_docs} documents "
        f"(avg_tokens={stats.avg_tokens:.1f}, median_tokens={stats.median_tokens:.0f}, "
        f"p95_tokens={stats.p95_tokens:.0f})"
    )

    embedded_chunks = embed_chunks(chunks)
    index = build_faiss_index(embedded_chunks)
    save_index(index, chunks)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()

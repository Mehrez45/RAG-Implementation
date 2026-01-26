from dataclasses import dataclass
from .pdf_loader import Document
from typing import List
import tiktoken

ENCODING_NAME = "cl100k_base"

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_number: int
    text: str
    start_token: int
    end_token: int

def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding(ENCODING_NAME)
    return len(encoder.encode(text))

def chunk_docs(documents: List[Document],
    max_tokens: int = 512,
    overlap_tokens: int = 0) -> List[Chunk]:
    """
    Splits documents into fixed-size token-aware chunks.
    """
    chunks: List[Chunk] = []
    encoder = tiktoken.get_encoding(ENCODING_NAME)
    if overlap_tokens >= max_tokens or overlap_tokens < 0:
        overlap_tokens = max_tokens // 10

    for doc in documents:
        tokens = encoder.encode(doc.text)

        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoder.decode(chunk_tokens)

            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_{doc.page_number}_{chunk_index}",
                    doc_id=doc.doc_id,
                    page_number=doc.page_number,
                    text=chunk_text,
                    start_token=start,
                    end_token=end
                ) 
            )
            if end == len(tokens):
                break
            start = end - overlap_tokens 
            chunk_index += 1
    return chunks

if __name__ == "__main__":
    from .pdf_loader import load_pdfs
    docs = load_pdfs("data/raw/pdfs")
    chunks = chunk_docs(docs, max_tokens=256, overlap_tokens= 40)

    print(f"Generated {len(chunks)} chunks")
    for chunk in chunks[:10]:
        print("\n---")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Tokens: {chunk.start_token}–{chunk.end_token}")
        print(chunk.text[:500])
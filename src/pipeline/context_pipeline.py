from typing import Optional
from src.llm.local_llm import LocalLLM
from src.retrieval.retriever import FaissRetriever
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.query_expander import QueryExpander
from src.retrieval.embeddings import embed_query
from src.ingestion.chunking import count_tokens

class ContextPipeline:

    def __init__(self, llm: LocalLLM, retriever: FaissRetriever,
                 k: int = 10, threshold: float = 0.35,
                 decomposer: Optional[QueryDecomposer] = None,
                 expander: Optional[QueryExpander] = None):
        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.threshold = threshold
        self.decomposer = decomposer
        self.expander = expander

    def run(self, query: str) -> list[str]:
        queries = [query]

        if self.decomposer is not None:
            queries = self.decomposer.decompose(query)
            if not queries:
                queries = [query]

        if self.expander is not None:
            expanded_queries = []
            for q in queries:
                expanded_queries.extend(self.expander.expand(q))
            if expanded_queries:
                queries = expanded_queries

        results = []
        for q in queries:
            query_vec = embed_query(q)
            result = self.retriever.retrieve_faiss(
                query_vec, k=self.k, threshold=self.threshold)
            results.extend(result)
        
        if not results:
            print("--- Context Extracted: 0 chunks found ---")
            return []

        seen = set()
        unique_chunks = []

        for chunk, score in sorted(results, key=lambda x: x[1], reverse=True):
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique_chunks.append(chunk)

        MAX_CONTEXT_TOKENS = 2500
        selected_chunks = []
        token_count = 0

        for chunk in unique_chunks:
            chunk_tokens = count_tokens(chunk.text)

            if token_count + chunk_tokens > MAX_CONTEXT_TOKENS:
                break

            selected_chunks.append(chunk.text)
            token_count += chunk_tokens

        print(f"--- Context Extracted: {len(selected_chunks)} chunks, {token_count} tokens ---")

        return selected_chunks
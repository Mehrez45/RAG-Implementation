from src.llm.local_llm import LocalLLM
from src.retrieval.retriever import FaissRetriever
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.query_expander import QueryExpander
from typing import Optional
from src.retrieval.embeddings import embed_query
from src.generation.rag_chain import build_rag_prompt

class RAGPipeline:
    def __init__(self, llm:LocalLLM, retriever: FaissRetriever,
                k: int = 10, threshold: float = 0.35,
                decomposer: Optional[QueryDecomposer] = None,
                expander: Optional[QueryExpander] = None):
        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.threshold = threshold
        self.decomposer = decomposer
        self.expander = expander

    def run(self, query: str) -> str:
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
            return "I don't know based on the provided context."

        seen = set()
        unique_chunks = []
        for chunk, score in results:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique_chunks.append(chunk)

        prompt = build_rag_prompt(query, [c.text for c in unique_chunks])
        return self.llm.generate(prompt)

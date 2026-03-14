from src.llm.local_llm import LocalLLM
from src.retrieval.retriever import FaissRetriever
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import CrossEncoderReranker
from typing import Optional
from src.generation.rag_chain import build_rag_prompt
from src.pipeline.retrieval_context import prepare_context

class RAGPipeline:
    def __init__(self, llm:LocalLLM, retriever: FaissRetriever,
                candidate_k: int = 24, final_k: int = 6,
                threshold: float = 0.35,
                decomposer: Optional[QueryDecomposer] = None,
                expander: Optional[QueryExpander] = None,
                reranker: Optional[CrossEncoderReranker] = None):
        self.llm = llm
        self.retriever = retriever
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.threshold = threshold
        self.decomposer = decomposer
        self.expander = expander
        self.reranker = reranker

    def run(self, query: str) -> str:
        context = prepare_context(
            query=query,
            retriever=self.retriever,
            candidate_k=self.candidate_k,
            final_k=self.final_k,
            threshold=self.threshold,
            decomposer=self.decomposer,
            expander=self.expander,
            reranker=self.reranker,
        )

        if not context.chunks:
            return "I don't know based on the provided context."

        MAX_CONTEXT_TOKENS = 2500
        selected_chunks = []
        token_count = 0

        for chunk in context.chunks:
            chunk_tokens = chunk.end_token - chunk.start_token

            if token_count + chunk_tokens > MAX_CONTEXT_TOKENS:
                break

            selected_chunks.append(chunk.text)
            token_count += chunk_tokens

        print(
            "RAG context:"
            f" {len(selected_chunks)} chunks, {token_count} tokens"
            f" | candidates={context.candidate_count}"
            f" | reranker={context.reranker_status}"
            f" | rerank_applied={context.reranker_applied}"
        )

        prompt = build_rag_prompt(query, selected_chunks)
        return self.llm.generate(prompt)

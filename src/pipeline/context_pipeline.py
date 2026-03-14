from dataclasses import dataclass
from typing import Optional

from src.llm.local_llm import LocalLLM
from src.pipeline.retrieval_context import prepare_ranking
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import FaissRetriever

MAX_CONTEXT_TOKENS = 2500
RETRY_CANDIDATE_STEP = 8
RETRY_THRESHOLD_STEP = 0.05
MIN_RETRIEVAL_THRESHOLD = 0.15


@dataclass
class ExtractedContext:
    texts: list[str]
    chunk_ids: list[str]
    candidate_count: int
    reranker_applied: bool
    reranker_status: str
    top_retrieval_score: Optional[float]
    top_rerank_score: Optional[float]


class ContextPipeline:
    def __init__(
        self,
        llm: LocalLLM,
        retriever: FaissRetriever,
        candidate_k: int = 24,
        final_k: int = 6,
        threshold: float = 0.35,
        decomposer: Optional[QueryDecomposer] = None,
        expander: Optional[QueryExpander] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.threshold = threshold
        self.decomposer = decomposer
        self.expander = expander
        self.reranker = reranker

    def run(self, query: str, attempt: int = 1) -> ExtractedContext:
        attempt = max(1, attempt)
        candidate_k = self.candidate_k + (attempt - 1) * RETRY_CANDIDATE_STEP
        threshold = max(
            MIN_RETRIEVAL_THRESHOLD,
            self.threshold - (attempt - 1) * RETRY_THRESHOLD_STEP,
        )

        ranking = prepare_ranking(
            query=query,
            retriever=self.retriever,
            candidate_k=candidate_k,
            threshold=threshold,
            decomposer=self.decomposer,
            expander=self.expander,
            reranker=self.reranker,
        )

        if not ranking.ranked_chunks:
            print("--- Context Extracted: 0 chunks found ---")
            return ExtractedContext(
                texts=[],
                chunk_ids=[],
                candidate_count=0,
                reranker_applied=ranking.reranker_applied,
                reranker_status=ranking.reranker_status,
                top_retrieval_score=None,
                top_rerank_score=None,
            )

        selected_texts: list[str] = []
        selected_chunk_ids: list[str] = []
        token_count = 0

        for item in ranking.ranked_chunks[: self.final_k]:
            chunk = item.chunk
            chunk_tokens = chunk.end_token - chunk.start_token

            if token_count + chunk_tokens > MAX_CONTEXT_TOKENS:
                break

            selected_texts.append(chunk.text)
            selected_chunk_ids.append(chunk.chunk_id)
            token_count += chunk_tokens

        top_result = ranking.ranked_chunks[0]

        print(
            "--- Context Extracted:"
            f" {len(selected_texts)} chunks, {token_count} tokens"
            f" | attempt={attempt}"
            f" | candidate_k={candidate_k}"
            f" | threshold={threshold:.2f}"
            f" | candidates={ranking.candidate_count}"
            f" | reranker={ranking.reranker_status}"
            f" | rerank_applied={ranking.reranker_applied} ---"
        )

        return ExtractedContext(
            texts=selected_texts,
            chunk_ids=selected_chunk_ids,
            candidate_count=ranking.candidate_count,
            reranker_applied=ranking.reranker_applied,
            reranker_status=ranking.reranker_status,
            top_retrieval_score=top_result.retrieval_score,
            top_rerank_score=top_result.rerank_score,
        )

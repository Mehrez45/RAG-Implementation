from dataclasses import dataclass
from typing import Optional

from src.ingestion.chunking import Chunk
from src.retrieval.embeddings import embed_query
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import CrossEncoderReranker, RankedChunk
from src.retrieval.retriever import FaissRetriever


@dataclass
class PreparedRanking:
    ranked_chunks: list[RankedChunk]
    candidate_count: int
    reranker_applied: bool
    reranker_status: str


@dataclass
class PreparedContext:
    chunks: list[Chunk]
    candidate_count: int
    reranker_applied: bool
    reranker_status: str


def _expand_queries(
    query: str,
    decomposer: Optional[QueryDecomposer],
    expander: Optional[QueryExpander],
) -> list[str]:
    queries = [query]

    if decomposer is not None:
        queries = decomposer.decompose(query)
        if not queries:
            queries = [query]

    if expander is not None:
        expanded_queries = []
        for candidate_query in queries:
            expanded_queries.extend(expander.expand(candidate_query))
        if expanded_queries:
            queries = expanded_queries

    return queries


def _dedupe_candidates(results: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    best_by_chunk_id: dict[str, tuple[Chunk, float]] = {}

    for chunk, score in results:
        current = best_by_chunk_id.get(chunk.chunk_id)
        if current is None or score > current[1]:
            best_by_chunk_id[chunk.chunk_id] = (chunk, score)

    return sorted(
        best_by_chunk_id.values(),
        key=lambda item: item[1],
        reverse=True,
    )


def prepare_context(
    query: str,
    retriever: FaissRetriever,
    candidate_k: int,
    final_k: int,
    threshold: float,
    decomposer: Optional[QueryDecomposer] = None,
    expander: Optional[QueryExpander] = None,
    reranker: Optional[CrossEncoderReranker] = None,
) -> PreparedContext:
    ranking = prepare_ranking(
        query=query,
        retriever=retriever,
        candidate_k=candidate_k,
        threshold=threshold,
        decomposer=decomposer,
        expander=expander,
        reranker=reranker,
    )

    return PreparedContext(
        chunks=[item.chunk for item in ranking.ranked_chunks[:final_k]],
        candidate_count=ranking.candidate_count,
        reranker_applied=ranking.reranker_applied,
        reranker_status=ranking.reranker_status,
    )


def prepare_ranking(
    query: str,
    retriever: FaissRetriever,
    candidate_k: int,
    threshold: float,
    decomposer: Optional[QueryDecomposer] = None,
    expander: Optional[QueryExpander] = None,
    reranker: Optional[CrossEncoderReranker] = None,
) -> PreparedRanking:
    queries = _expand_queries(query, decomposer=decomposer, expander=expander)

    results: list[tuple[Chunk, float]] = []
    for candidate_query in queries:
        query_vec = embed_query(candidate_query)
        results.extend(
            retriever.retrieve_faiss(
                query_vec,
                k=candidate_k,
                threshold=threshold,
            )
        )

    if not results:
        reranker_status = "disabled" if reranker is None else reranker.status
        return PreparedRanking(
            ranked_chunks=[],
            candidate_count=0,
            reranker_applied=False,
            reranker_status=reranker_status,
        )

    unique_candidates = _dedupe_candidates(results)[:candidate_k]

    if reranker is None:
        ranked_chunks = [
            RankedChunk(chunk=chunk, retrieval_score=score)
            for chunk, score in unique_candidates
        ]
        return PreparedRanking(
            ranked_chunks=ranked_chunks,
            candidate_count=len(unique_candidates),
            reranker_applied=False,
            reranker_status="disabled",
        )

    reranked = reranker.rerank(
        query=query,
        candidates=unique_candidates,
        top_n=None,
    )
    return PreparedRanking(
        ranked_chunks=reranked.chunks,
        candidate_count=len(unique_candidates),
        reranker_applied=reranked.applied,
        reranker_status=reranked.model_status,
    )

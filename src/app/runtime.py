import os
from dataclasses import dataclass
from typing import Optional

from src.llm.local_llm import LocalLLM
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import FaissRetriever
from src.retrieval.storage import load_index
from src.utilities.document_aliases import DocumentAliasCatalog, build_document_alias_catalog

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATE_K = 24
DEFAULT_FINAL_K = 6
DEFAULT_THRESHOLD = 0.35
DEFAULT_AGENTIC_MAX_REVISIONS = 2
DEFAULT_AGENTIC_REVIEWER_ACCEPT_SCORE = 3.0


@dataclass
class AppRuntime:
    llm: LocalLLM
    retriever: FaissRetriever
    expander: Optional[QueryExpander]
    decomposer: Optional[QueryDecomposer]
    reranker: Optional[CrossEncoderReranker]
    candidate_k: int
    final_k: int
    threshold: float
    max_revisions: int
    reviewer_accept_score: float
    force_retrieval: bool
    document_aliases: DocumentAliasCatalog


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() not in {"0", "false", "no", "off"}


def _read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def _read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def build_runtime() -> AppRuntime:
    llm = LocalLLM()
    index, chunks = load_index()
    retriever = FaissRetriever(index, chunks)
    document_aliases = build_document_alias_catalog(chunks)
    expander = QueryExpander(llm=llm)
    decomposer = QueryDecomposer(llm=llm)
    threshold = _read_float_env("RAG_RETRIEVAL_THRESHOLD", DEFAULT_THRESHOLD)
    final_k = max(1, _read_int_env("RAG_FINAL_K", DEFAULT_FINAL_K))
    candidate_k = max(final_k, _read_int_env("RAG_CANDIDATE_K", DEFAULT_CANDIDATE_K))
    max_revisions = max(
        1,
        _read_int_env("RAG_AGENTIC_MAX_REVISIONS", DEFAULT_AGENTIC_MAX_REVISIONS),
    )
    reviewer_accept_score = _read_float_env(
        "RAG_AGENTIC_REVIEWER_ACCEPT_SCORE",
        DEFAULT_AGENTIC_REVIEWER_ACCEPT_SCORE,
    )
    force_retrieval = _read_bool_env("RAG_AGENTIC_FORCE_RETRIEVAL", False)

    reranker = None
    if _read_bool_env("RAG_ENABLE_RERANKER", True):
        reranker = CrossEncoderReranker(
            model_name=os.getenv("RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
        )

    return AppRuntime(
        llm=llm,
        retriever=retriever,
        expander=expander,
        decomposer=decomposer,
        reranker=reranker,
        candidate_k=candidate_k,
        final_k=final_k,
        threshold=threshold,
        max_revisions=max_revisions,
        reviewer_accept_score=reviewer_accept_score,
        force_retrieval=force_retrieval,
        document_aliases=document_aliases,
    )

from typing import List, Optional, TypedDict

class AgentState(TypedDict):
    user_query: str
    extracted_facts: List[str]
    extracted_chunk_ids: List[str]
    current_draft: str
    review_feedback: str
    is_relevant: bool
    revision_count: int
    failure_reason: str
    needs_retrieval: bool
    document_grounded: bool
    route: str
    route_reason: str
    candidate_count: int
    top_retrieval_score: Optional[float]
    top_rerank_score: Optional[float]
    reranker_applied: bool
    retrieval_stalled: bool

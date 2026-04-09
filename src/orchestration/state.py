from typing import List, TypedDict

class AgentState(TypedDict):
    user_query: str
    extracted_facts: List[str]
    current_draft: str
    review_feedback: str
    is_relevant: bool
    revision_count: int
    failure_reason: str
    needs_retrieval: bool
    route: str
    route_reason: str

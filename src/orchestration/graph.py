from typing import Optional

from langgraph.graph import END, StateGraph

from src.agents.direct_responder import build_direct_responder_node
from src.agents.extractor import build_extractor_node
from src.agents.reviewer import build_reviewer_node
from src.agents.router import build_router_node
from src.agents.summarizer import build_summarizer_node
from src.llm.local_llm import LocalLLM
from src.orchestration.state import AgentState
from src.utilities.document_aliases import DocumentAliasCatalog


def create_rag_graph(
    context_pipeline,
    llm: LocalLLM,
    *,
    max_revisions: int = 2,
    force_retrieval: bool = False,
    reviewer_accept_score: float = 3.0,
    document_aliases: Optional[DocumentAliasCatalog] = None,
):
    workflow = StateGraph(AgentState)

    workflow.add_node(
        "router",
        build_router_node(
            llm,
            force_retrieval=force_retrieval,
            document_aliases=document_aliases,
        ),
    )
    workflow.add_node("direct_responder", build_direct_responder_node(llm))
    workflow.add_node("extractor", build_extractor_node(context_pipeline))
    workflow.add_node(
        "reviewer",
        build_reviewer_node(llm, short_circuit_rerank_score=reviewer_accept_score),
    )
    workflow.add_node("summarizer", build_summarizer_node(llm))

    def decide_retrieval_path(state: AgentState) -> str:
        if state.get("needs_retrieval", True):
            return "retrieve"

        return "direct"

    def decide_after_review(state: AgentState) -> str:
        if state.get("is_relevant", False):
            return "sufficient"

        if state.get("retrieval_stalled", False):
            if state.get("document_grounded", False):
                return "force_generate"
            return "fallback_direct"

        if state.get("revision_count", 0) >= max_revisions:
            if state.get("document_grounded", False):
                return "force_generate"
            return "fallback_direct"

        return "insufficient"

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        decide_retrieval_path,
        {
            "direct": "direct_responder",
            "retrieve": "extractor",
        },
    )

    workflow.add_edge("direct_responder", END)
    workflow.add_edge("extractor", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        decide_after_review,
        {
            "sufficient": "summarizer",
            "insufficient": "extractor",
            "force_generate": "summarizer",
            "fallback_direct": "direct_responder",
        },
    )

    workflow.add_edge("summarizer", END)
    return workflow.compile()

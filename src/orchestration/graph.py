from langgraph.graph import StateGraph, END
from src.orchestration.state import AgentState
from src.agents.direct_responder import build_direct_responder_node
from src.agents.extractor import build_extractor_node
from src.agents.router import build_router_node
from src.agents.reviewer import build_reviewer_node
from src.agents.summarizer import build_summarizer_node
from src.llm.local_llm import LocalLLM


MAX_REVISIONS = 3


def create_rag_graph(context_pipeline, llm: LocalLLM):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", build_router_node(llm))
    workflow.add_node("direct_responder", build_direct_responder_node(llm))
    workflow.add_node("extractor", build_extractor_node(context_pipeline))
    workflow.add_node("reviewer", build_reviewer_node(llm))
    workflow.add_node("summarizer", build_summarizer_node(llm))

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
        decide_to_generate,
        {
            "sufficient": "summarizer",
            "insufficient": "extractor",
            "force_generate": "summarizer",
        },
    )

    workflow.add_edge("summarizer", END)
    return workflow.compile()


def decide_retrieval_path(state: AgentState) -> str:
    if state.get("needs_retrieval", True):
        return "retrieve"

    return "direct"


def decide_to_generate(state: AgentState) -> str:
    if state.get("is_relevant", False):
        return "sufficient"

    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "force_generate"

    return "insufficient"

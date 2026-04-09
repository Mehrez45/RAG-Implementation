from src.llm.local_llm import LocalLLM
from src.orchestration.state import AgentState


DIRECT_QUERIES = {
    "good afternoon",
    "good evening",
    "good morning",
    "hello",
    "hey",
    "hi",
    "how are you",
    "thank you",
    "thanks",
    "whats up",
    "what's up",
}


def build_router_node(llm: LocalLLM):
    def run_router(state: AgentState) -> dict:
        print("--- ROUTER AGENT: Deciding Whether Retrieval Is Needed ---")
        query = " ".join(state["user_query"].strip().split())
        normalized_query = query.lower()

        if normalized_query in DIRECT_QUERIES:
            return {
                "needs_retrieval": False,
                "route": "direct",
                "route_reason": "heuristic_small_talk",
            }

        prompt = f"""
        You are routing a user query in a local assistant.

        Return exactly one label:
        - DIRECT: the query is casual conversation or stable general knowledge
          that can be answered without retrieved documents.
        - RETRIEVE: the query needs the ingested documents, project-specific
          knowledge, or you are not fully sure.

        Examples:
        Query: hello
        Label: DIRECT

        Query: why is the sky blue?
        Label: DIRECT

        Query: summarize the uploaded documents
        Label: RETRIEVE

        Query: what does the paper say about retrieval?
        Label: RETRIEVE

        Query: {query}
        Label:
        """

        label = llm.generate(prompt, max_tokens=4, temperature=0.0).strip().upper()
        is_direct = label.startswith("DIRECT")

        return {
            "needs_retrieval": not is_direct,
            "route": "direct" if is_direct else "retrieve",
            "route_reason": f"router_label={label or 'RETRIEVE'}",
        }

    return run_router

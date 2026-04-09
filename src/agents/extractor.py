from src.orchestration.state import AgentState

def build_extractor_node(context_pipeline):

    def run_extractor(state: AgentState) -> dict:
        print("--- EXTRACTOR AGENT: Searching the Vector DB ---")
        question = state["user_query"]
        retrieved_chunks = context_pipeline.run(question)
        current_revision = state.get("revision_count", 0)

        return {
            "extracted_facts": retrieved_chunks,
            "revision_count": current_revision + 1,
        }

    return run_extractor

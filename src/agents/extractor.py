from src.orchestration.state import AgentState


def build_extractor_node(context_pipeline):

    def run_extractor(state: AgentState) -> dict:
        print("--- EXTRACTOR AGENT: Searching the Vector DB ---")
        question = state["user_query"]
        current_revision = state.get("revision_count", 0)
        next_attempt = current_revision + 1
        previous_chunk_ids = state.get("extracted_chunk_ids", [])
        extracted = context_pipeline.run(question, attempt=next_attempt)
        retrieval_stalled = bool(previous_chunk_ids) and (
            extracted.chunk_ids == previous_chunk_ids
        )

        return {
            "extracted_facts": extracted.texts,
            "extracted_chunk_ids": extracted.chunk_ids,
            "revision_count": next_attempt,
            "candidate_count": extracted.candidate_count,
            "top_retrieval_score": extracted.top_retrieval_score,
            "top_rerank_score": extracted.top_rerank_score,
            "reranker_applied": extracted.reranker_applied,
            "retrieval_stalled": retrieval_stalled,
        }

    return run_extractor

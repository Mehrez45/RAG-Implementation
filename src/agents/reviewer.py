from src.orchestration.state import AgentState
from src.llm.local_llm import LocalLLM

MAX_REVIEW_CHUNKS = 2
MAX_REVIEW_CHARS_PER_CHUNK = 700


def _parse_review_label(raw_response: str) -> bool:
    normalized = " ".join(raw_response.strip().upper().split())
    if not normalized:
        return False

    yes_index = normalized.find("YES")
    no_index = normalized.find("NO")
    candidates: list[tuple[int, bool]] = []
    if yes_index != -1:
        candidates.append((yes_index, True))
    if no_index != -1:
        candidates.append((no_index, False))

    if candidates:
        return min(candidates, key=lambda item: item[0])[1]

    for character in normalized:
        if character == "Y":
            return True
        if character == "N":
            return False

    return False


def build_reviewer_node(
    llm: LocalLLM,
    short_circuit_rerank_score: float = 3.0,
):
    def run_reviewer(state: AgentState) -> dict:
        print("--- REVIEWER AGENT: Validating Context ---")
        chunks = state.get("extracted_facts", [])
        query = state["user_query"]
        top_rerank_score = state.get("top_rerank_score")

        if not chunks:
            return {
                "is_relevant": False,
                "review_feedback": "No context chunks were retrieved for this question.",
            }

        if state.get("retrieval_stalled", False):
            return {
                "is_relevant": False,
                "review_feedback": (
                    "Retrieval returned the same top chunks again, so the agent "
                    "is stopping additional retries."
                ),
            }

        if (
            state.get("reranker_applied", False)
            and top_rerank_score is not None
            and top_rerank_score >= short_circuit_rerank_score
        ):
            return {
                "is_relevant": True,
                "review_feedback": (
                    "Accepted high-confidence retrieval without an additional review "
                    f"pass (top rerank score={top_rerank_score:.2f})."
                ),
            }

        review_chunks = [
            chunk[:MAX_REVIEW_CHARS_PER_CHUNK] for chunk in chunks[:MAX_REVIEW_CHUNKS]
        ]
        context_text = "\n\n".join(review_chunks)
        prompt = f"""
        You are checking whether retrieved context is sufficient to answer a question.

        Return exactly one label:
        Y
        N

        Y means the retrieved text is relevant and likely contains enough information
        to answer the question.
        N means the text is missing the answer or is off-topic.

        Question: {query}
        Retrieved text:
        {context_text}

        Label:
        """

        response = llm.generate(
            prompt,
            max_tokens=2,
            temperature=0.0,
            stop=["\n", "END", "</s>"],
        )
        is_relevant = _parse_review_label(response)

        if is_relevant:
            feedback = "Retrieved context is relevant to the question."
        else:
            feedback = "Retrieved context is not relevant enough to answer the question."

        return {
            "is_relevant": is_relevant,
            "review_feedback": feedback,
        }

    return run_reviewer

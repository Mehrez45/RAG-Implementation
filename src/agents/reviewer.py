from src.orchestration.state import AgentState
from src.llm.local_llm import LocalLLM


def build_reviewer_node(llm: LocalLLM):
    def run_reviewer(state: AgentState) -> dict:
        print("--- REVIEWER AGENT: Validating Context ---")
        chunks = state.get("extracted_facts", [])
        query = state["user_query"]

        if not chunks:
            return {
                "is_relevant": False,
                "review_feedback": "No context chunks were retrieved for this question.",
            }

        context_text = "\n\n".join(chunks)
        prompt = f"""
        You are a grader assessing relevance of a retrieved document to a user question. 
        Retrieved Document: {context_text}
        User Question: {query}
        
        If the document contains keywords or semantic meaning related to the user question, grade it as relevant. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        """

        response = llm.generate(prompt).strip().lower()

        if response.startswith("yes"):
            is_relevant = True
            feedback = "Retrieved context is relevant to the question."
        else:
            is_relevant = False
            feedback = "Retrieved context is not relevant enough to answer the question."

        return {
            "is_relevant": is_relevant,
            "review_feedback": feedback,
        }

    return run_reviewer

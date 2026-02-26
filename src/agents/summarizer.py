from src.generation.rag_chain import build_rag_prompt
from src.llm.local_llm import LocalLLM
from src.orchestration.state import AgentState


def build_summarizer_node(llm: LocalLLM):
    def run_summarizer(state: AgentState) -> dict:
        print("--- SUMMARIZER AGENT: Generating Answer ---")

        query = state["user_query"]
        contexts = state.get("extracted_facts", [])

        if not contexts:
            return {
                "current_draft": "I don't know based on the provided context.",
                "failure_reason": "No retrieved context was available for summarization.",
            }

        prompt = build_rag_prompt(query, contexts)
        answer = llm.generate(prompt)

        return {"current_draft": answer}

    return run_summarizer

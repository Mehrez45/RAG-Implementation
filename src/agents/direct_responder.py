from src.generation.direct_chain import build_direct_prompt
from src.llm.local_llm import LocalLLM
from src.orchestration.state import AgentState


def build_direct_responder_node(llm: LocalLLM):
    def run_direct_responder(state: AgentState) -> dict:
        print("--- DIRECT RESPONDER AGENT: Generating Answer Without Retrieval ---")

        query = state["user_query"]
        prompt = build_direct_prompt(query)
        answer = llm.generate(prompt)

        return {
            "current_draft": answer,
            "failure_reason": "",
        }

    return run_direct_responder

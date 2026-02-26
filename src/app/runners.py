from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from src.app.runtime import AppRuntime
from src.orchestration.graph import create_rag_graph
from src.orchestration.state import AgentState
from src.pipeline.context_pipeline import ContextPipeline
from src.pipeline.rag_pipeline import RAGPipeline

RunnerMode = Literal["vanilla", "agentic"]


@dataclass
class RunResult:
    mode: RunnerMode
    answer: str
    revision_count: int = 0
    review_feedback: str = ""
    failure_reason: str = ""


class QueryRunner(ABC):
    mode: RunnerMode

    @abstractmethod
    def run(self, query: str) -> RunResult:
        raise NotImplementedError


class VanillaRunner(QueryRunner):
    mode: RunnerMode = "vanilla"

    def __init__(self, runtime: AppRuntime):
        self.pipeline = RAGPipeline(
            llm=runtime.llm,
            retriever=runtime.retriever,
            expander=runtime.expander,
            decomposer=runtime.decomposer,
        )

    def run(self, query: str) -> RunResult:
        answer = self.pipeline.run(query)
        return RunResult(mode=self.mode, answer=answer)


class AgenticRunner(QueryRunner):
    mode: RunnerMode = "agentic"

    def __init__(self, runtime: AppRuntime):
        context_pipeline = ContextPipeline(
            llm=runtime.llm,
            retriever=runtime.retriever,
            expander=runtime.expander,
            decomposer=runtime.decomposer,
        )
        self.graph = create_rag_graph(context_pipeline, runtime.llm)

    def run(self, query: str) -> RunResult:
        initial_state: AgentState = {
            "user_query": query,
            "extracted_facts": [],
            "current_draft": "",
            "review_feedback": "",
            "is_relevant": False,
            "revision_count": 0,
            "failure_reason": "",
        }

        final_state = self.graph.invoke(initial_state)
        answer = final_state.get(
            "current_draft",
            "I don't know based on the provided context.",
        )

        return RunResult(
            mode=self.mode,
            answer=answer,
            revision_count=final_state.get("revision_count", 0),
            review_feedback=final_state.get("review_feedback", ""),
            failure_reason=final_state.get("failure_reason", ""),
        )


def build_runner(mode: RunnerMode, runtime: AppRuntime) -> QueryRunner:
    if mode == "agentic":
        return AgenticRunner(runtime)

    return VanillaRunner(runtime)

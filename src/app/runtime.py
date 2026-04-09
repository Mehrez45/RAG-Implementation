from dataclasses import dataclass

from src.llm.local_llm import LocalLLM
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.query_expander import QueryExpander
from src.retrieval.retriever import FaissRetriever
from src.retrieval.storage import load_index


@dataclass
class AppRuntime:
    llm: LocalLLM
    retriever: FaissRetriever
    expander: QueryExpander
    decomposer: QueryDecomposer


def build_runtime() -> AppRuntime:
    llm = LocalLLM()
    index, chunks = load_index()
    retriever = FaissRetriever(index, chunks)
    expander = QueryExpander(llm=llm)
    decomposer = QueryDecomposer(llm=llm)

    return AppRuntime(
        llm=llm,
        retriever=retriever,
        expander=expander,
        decomposer=decomposer,
    )

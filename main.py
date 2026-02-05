from src.llm.local_llm import LocalLLM
from src.retrieval.retriever import FaissRetriever
from src.retrieval.storage import load_index
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.query_expander import QueryExpander
from src.retrieval.query_decomposer import QueryDecomposer

def main():
    print("Loading components...")

    llm = LocalLLM()
    index, chunks = load_index()
    retriever = FaissRetriever(index, chunks)
    expander = QueryExpander(llm=llm)
    decomposer = QueryDecomposer(llm=llm)

    pipeline = RAGPipeline(
        llm=llm, retriever=retriever,
        expander=expander, decomposer=decomposer)

    print("Ready.\nType quit() in order to exit")
    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question == "quit()":
            break

        answer = pipeline.run(question)
        print("\n", answer, "\n")


if __name__ == "__main__":
    main()

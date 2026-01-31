from src.llm.local_llm import LocalLLM
from src.generation.rag_chain import build_rag_prompt
from src.retrieval.embeddings import embed_query
from src.retrieval.retriever import retrieve_faiss
from src.retrieval.storage import load_index



def main():
    print("Loading components...")

    llm = LocalLLM()
    index, chunks = load_index()

    print("Ready.\nType quit() in order to exit")
    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question == "quit()":
            break

        query_vec = embed_query(question)

        results = retrieve_faiss(
            query_vec=query_vec,
            index=index,
            chunks=chunks,
            k=10,
            threshold=0.35
        )
        print("\n--- RETRIEVED CHUNKS ---")
        for chunk, score in results:
            print(f"[score={score:.3f}] {chunk.text[:200]}")
        print("------------------------\n")


        if not results:
            print("I don’t know based on the provided context.")
            continue

        contexts = [chunk.text for chunk, score in results]

        prompt = build_rag_prompt(question, contexts)

        answer = llm.generate(prompt)
        print("\n", answer, "\n")



if __name__ == "__main__":
    main()

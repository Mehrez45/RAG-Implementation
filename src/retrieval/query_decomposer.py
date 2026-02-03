from llm.local_llm import LocalLLM

class QueryDecomposer:
    def __init__(self, llm: LocalLLM):
        self.llm = llm

    def decompose(self, query):
        prompt = f"""
Decompose the following user question into smaller standalone search queries if necessary.
If the question is simple, return the original question only.
User question:
{query}
Return each query on a new line
"""
        output = self.llm.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=128
        )

        queries = [q.strip() for q in output.split("\n") if q.strip()]
        return queries
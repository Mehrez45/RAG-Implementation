from llm.local_llm import LocalLLM

class QueryExpander:
    def __init__(self, llm: LocalLLM, num_queries=5):
        self.llm = llm
        self.num_queries = num_queries

    def rewrite(self, query: str):
        prompt = f"""
Generate {self.num_queries} different search queries that could retrieve relevant technical documents for the following user question.
User question:
{query}

Return each query on a new line.
"""
        output = self.llm.generate(
            prompt=prompt,
            max_tokens=128,
            temperature=0.15
        )

        queries = [q.strip() for q in output.split("\n") if q.strip()]
        return queries
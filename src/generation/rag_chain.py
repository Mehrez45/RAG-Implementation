SYSTEM_PROMPT = """You are a retrieval-based assistant.

Use the provided context to answer the question.
Do not use information that is not supported by the context.
Do not speculate or invent facts.

If the answer is NOT explicitly stated or directly supported by the context, respond EXACTLY with:
"I don’t know based on the provided context."

Answer ONCE.
Do not repeat or restate your answer.
Be concise and factual.
"""


def build_rag_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)

    return f"""{SYSTEM_PROMPT}

Context:
{context_block}

Question:
{question}

Answer:
"""

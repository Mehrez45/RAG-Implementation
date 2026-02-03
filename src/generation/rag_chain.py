SYSTEM_PROMPT = """You are a retrieval-based assistant.

You must answer the question using ONLY the information in the context below.
Do NOT use prior knowledge.
Do NOT make assumptions.
Do NOT invent facts.
Do NOT repeat yourself
Do utilise initiative

If the answer is not explicitly stated in the context, respond with:
"I don’t know based on the provided context."
"""

def build_rag_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)

    return f"""{SYSTEM_PROMPT}

Context:
{context_block}

Question:
{question}

Answer (concise, factual, no filler):
"""

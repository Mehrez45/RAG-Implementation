from typing import Optional

from src.llm.local_llm import LocalLLM
from src.orchestration.state import AgentState
from src.utilities.document_aliases import DocumentAliasCatalog, normalize_routing_text


DIRECT_QUERIES = {
    "good afternoon", "good evening", "good morning", "hello", "hey",
    "hi", "how are you", "thank you", "thanks", "whats up", "what's up",
}

LOCAL_DOCUMENT_HINTS = (
    "uploaded document", "uploaded documents", "uploaded pdf", "uploaded file",
    "what does this paper", "what does this article", "what does the document", "what does the pdf",
    "in this paper", "in this article", "in the document", "in the pdf",
    "from this paper",
    "from this article",
    "from the document",
    "from the pdf",
    "the uploaded",
    "these docs",
    "the docs",
    "based on the context",
    "using the context",
)

SOURCE_REFERENCE_HINTS = (
    "paper",
    "article",
    "document",
    "pdf",
    "report",
    "study",
    "research",
    "authors",
    "title",
    "section",
)

DOCUMENT_TOPIC_HINTS = (
    "compare ",
    "describe ",
    "explain ",
    "how does ",
    "summarise ",
    "summarize ",
    "tell me about ",
    "what does ",
    "what is ",
    "who wrote ",
)

MAX_ROUTER_ALIAS_EXAMPLES = 20


def looks_local_document_grounded(normalized_query: str) -> bool:
    return any(hint in normalized_query for hint in LOCAL_DOCUMENT_HINTS)


def _parse_router_label(raw_label: str) -> str:
    normalized = " ".join(raw_label.strip().upper().split())
    if not normalized:
        return "RETRIEVE"

    direct_index = normalized.find("DIRECT")
    retrieve_index = normalized.find("RETRIEVE")
    partial_retrieve_index = normalized.find("RETRIE")

    candidates: list[tuple[int, str]] = []
    if direct_index != -1:
        candidates.append((direct_index, "DIRECT"))
    if retrieve_index != -1:
        candidates.append((retrieve_index, "RETRIEVE"))
    elif partial_retrieve_index != -1:
        candidates.append((partial_retrieve_index, "RETRIEVE"))

    if candidates:
        return min(candidates, key=lambda item: item[0])[1]

    for character in normalized:
        if character == "D":
            return "DIRECT"
        if character == "R":
            return "RETRIEVE"

    return "RETRIEVE"


def _contains_alias(normalized_query: str, alias: str) -> bool:
    padded_query = f" {normalized_query} "
    padded_alias = f" {alias} "
    return padded_alias in padded_query


def _matching_aliases(
    normalized_query: str,
    aliases: tuple[str, ...],
) -> list[str]:
    return [alias for alias in aliases if _contains_alias(normalized_query, alias)]


def _has_source_reference_cue(normalized_query: str) -> bool:
    return any(_contains_alias(normalized_query, cue) for cue in SOURCE_REFERENCE_HINTS)


def _looks_document_topic_query(normalized_query: str) -> bool:
    return any(normalized_query.startswith(prefix) for prefix in DOCUMENT_TOPIC_HINTS)


def _format_alias_examples(catalog: DocumentAliasCatalog) -> str:
    examples = list(catalog.phrase_aliases[:MAX_ROUTER_ALIAS_EXAMPLES])
    remaining_slots = MAX_ROUTER_ALIAS_EXAMPLES - len(examples)
    if remaining_slots > 0:
        examples.extend(catalog.keyword_aliases[:remaining_slots])

    if not examples:
        return "- no indexed titles loaded"

    return "\n".join(f"- {alias}" for alias in examples)


def build_router_node(
    llm: LocalLLM,
    force_retrieval: bool = False,
    document_aliases: Optional[DocumentAliasCatalog] = None,
):
    catalog = document_aliases or DocumentAliasCatalog(phrase_aliases=(), keyword_aliases=())
    alias_examples = _format_alias_examples(catalog)

    def run_router(state: AgentState) -> dict:
        print("--- ROUTER AGENT: Deciding Whether Retrieval Is Needed ---")
        query = " ".join(state["user_query"].strip().split())
        normalized_query = normalize_routing_text(query)
        matched_phrases = _matching_aliases(normalized_query, catalog.phrase_aliases)
        matched_keywords = _matching_aliases(normalized_query, catalog.keyword_aliases)

        if force_retrieval:
            return {
                "needs_retrieval": True,
                "document_grounded": True,
                "route": "retrieve",
                "route_reason": "forced_retrieval",
            }

        if normalized_query in DIRECT_QUERIES:
            return {
                "needs_retrieval": False,
                "document_grounded": False,
                "route": "direct",
                "route_reason": "heuristic_small_talk",
            }

        if looks_local_document_grounded(normalized_query):
            return {
                "needs_retrieval": True,
                "document_grounded": True,
                "route": "retrieve",
                "route_reason": "heuristic_local_document_query",
            }

        if matched_phrases:
            matched_title = max(matched_phrases, key=len)
            return {
                "needs_retrieval": True,
                "document_grounded": True,
                "route": "retrieve",
                "route_reason": f"heuristic_indexed_title={matched_title}",
            }

        if matched_keywords and (
            _has_source_reference_cue(normalized_query)
            or _looks_document_topic_query(normalized_query)
        ):
            matched_title = max(matched_keywords, key=len)
            return {
                "needs_retrieval": True,
                "document_grounded": True,
                "route": "retrieve",
                "route_reason": f"heuristic_indexed_keyword={matched_title}",
            }

        prompt = f"""
        You are routing a user query in a local assistant.

        Return exactly one label:
        D
        R

        D means the query is casual conversation or stable general knowledge
        that can be answered without retrieved documents.
        R means the query needs the ingested documents, project-specific
        knowledge, or you are not fully sure.

        Indexed local document titles and aliases:
        {alias_examples}

        Prefer R when the query mentions one of the indexed titles,
        a likely title keyword plus a source cue like paper/article/document,
        or when the user appears to want evidence grounded in the local corpus.
        Prefer D for casual chat, broad general knowledge, or references
        to public sources that are not clearly part of the indexed documents.

        Examples:
        Query: hello
        Label: D

        Query: why is the sky blue?
        Label: D

        Query: summarize the uploaded documents
        Label: R

        Query: what does the paper say about retrieval?
        Label: R

        Query: what is longformer?
        Label: R

        Query: {query}
        Label:
        """

        raw_label = llm.generate(
            prompt,
            max_tokens=2,
            temperature=0.0,
            stop=["\n", "END", "</s>"],
        )
        parsed_label = _parse_router_label(raw_label)
        is_direct = parsed_label == "DIRECT"

        return {
            "needs_retrieval": not is_direct,
            "document_grounded": False,
            "route": "direct" if is_direct else "retrieve",
            "route_reason": f"router_label={parsed_label}",
        }

    return run_router

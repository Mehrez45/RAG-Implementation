import re
from dataclasses import dataclass
from typing import Iterable

from src.ingestion.chunking import Chunk

CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z][a-z])")
LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
NON_ALNUM = re.compile(r"[^a-z0-9]+")

EDGE_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "using",
}

TITLE_LINE_BLOCKLIST = (
    "provided proper attribution",
    "journalistic",
    "scholarly works",
    "join us",
    "all rights reserved",
    "copyright",
)

RAW_LINE_BLOCKLIST = (
    "@",
    ".com",
    "in collaboration with others",
    "university",
    "institute",
    "department",
    "school",
    "college",
    "facebook",
    "google",
    "allen institute",
    "new york",
    "seattle",
)

TITLE_STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "using",
    "without",
    "you",
}


@dataclass(frozen=True)
class DocumentAliasCatalog:
    phrase_aliases: tuple[str, ...]
    keyword_aliases: tuple[str, ...]


def normalize_routing_text(text: str) -> str:
    prepared = CAMEL_BOUNDARY.sub(" ", text)
    prepared = LETTER_DIGIT_BOUNDARY.sub(" ", prepared)
    prepared = prepared.replace("_", " ").replace("-", " ")
    return " ".join(NON_ALNUM.sub(" ", prepared.lower()).split())


def _normalize_simple_text(text: str) -> str:
    prepared = LETTER_DIGIT_BOUNDARY.sub(" ", text)
    prepared = prepared.replace("_", " ").replace("-", " ")
    return " ".join(NON_ALNUM.sub(" ", prepared.lower()).split())


def _normalized_variants(text: str) -> set[str]:
    variants = {
        normalize_routing_text(text),
        _normalize_simple_text(text),
    }
    return {variant for variant in variants if variant}


def _trim_edge_stopwords(text: str) -> str:
    words = text.split()
    while words and words[0] in EDGE_STOPWORDS:
        words = words[1:]
    while words and words[-1] in EDGE_STOPWORDS:
        words = words[:-1]
    return " ".join(words)


def _candidate_title_lines(text: str, doc_id: str) -> list[str]:
    candidates: list[tuple[float, int, str]] = []
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    doc_tokens = {
        token
        for variant in _normalized_variants(doc_id)
        for token in variant.split()
        if len(token) >= 3
    }

    for line_index, line in enumerate(raw_lines[:12]):
        raw_lower = line.lower()
        if any(blocked in raw_lower for blocked in RAW_LINE_BLOCKLIST):
            continue

        normalized_variants = _normalized_variants(line)
        normalized = max(normalized_variants, key=len, default="")
        if not normalized or len(normalized) < 6:
            continue
        if any(blocked in normalized for blocked in TITLE_LINE_BLOCKLIST):
            continue

        word_count = len(normalized.split())
        if not 2 <= word_count <= 16:
            continue

        comma_penalty = raw_lower.count(",") * 1.5
        base_score = 1.0 if 3 <= word_count <= 10 and raw_lower.count(",") == 0 else 0.0
        overlap_score = max(
            (
                len(doc_tokens & set(variant.split()))
                for variant in normalized_variants
            ),
            default=0,
        )
        stopword_score = sum(
            1 for word in normalized.split() if word in TITLE_STOPWORDS
        )
        title_score = (
            base_score
            + overlap_score * 4
            + min(stopword_score, 3)
            - comma_penalty
            - max(0, word_count - 14) * 0.5
            - line_index * 0.2
        )
        if title_score < 1.5:
            continue

        for variant in normalized_variants:
            trimmed = _trim_edge_stopwords(variant)
            if trimmed and 2 <= len(trimmed.split()) <= 16:
                candidates.append((title_score, line_index, trimmed))

    candidates.sort(key=lambda item: (-item[0], item[1], -len(item[2])))

    selected: list[str] = []
    seen: set[str] = set()
    for _, _, candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        selected.append(candidate)
        if len(selected) >= 1:
            break

    return selected


def build_document_alias_catalog(chunks: Iterable[Chunk]) -> DocumentAliasCatalog:
    earliest_chunk_by_doc: dict[str, Chunk] = {}

    for chunk in chunks:
        current = earliest_chunk_by_doc.get(chunk.doc_id)
        if current is None or (
            chunk.page_number,
            chunk.start_token,
        ) < (
            current.page_number,
            current.start_token,
        ):
            earliest_chunk_by_doc[chunk.doc_id] = chunk

    phrase_aliases: set[str] = set()
    keyword_aliases: set[str] = set()

    for doc_id, chunk in earliest_chunk_by_doc.items():
        aliases = _normalized_variants(doc_id)
        aliases.update(_candidate_title_lines(chunk.text, doc_id))
        best_alias_by_compact: dict[str, str] = {}

        for alias in aliases:
            cleaned = _trim_edge_stopwords(alias)
            if not cleaned or len(cleaned) < 4:
                continue

            compact = cleaned.replace(" ", "")
            current = best_alias_by_compact.get(compact)
            if current is None or len(cleaned.split()) > len(current.split()):
                best_alias_by_compact[compact] = cleaned

        for cleaned in best_alias_by_compact.values():
            if len(cleaned.split()) >= 2:
                phrase_aliases.add(cleaned)
            else:
                keyword_aliases.add(cleaned)

    return DocumentAliasCatalog(
        phrase_aliases=tuple(sorted(phrase_aliases)),
        keyword_aliases=tuple(sorted(keyword_aliases)),
    )

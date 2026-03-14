import re
from typing import Iterable

TEXT_NORMALIZER = re.compile(r"[^a-z0-9]+")

PatternSlot = list[str]
PatternGroup = list[PatternSlot]


def normalize_text(text: str) -> str:
    return " ".join(TEXT_NORMALIZER.sub(" ", text.lower()).split())


def parse_answer_patterns(raw_patterns: Iterable[object]) -> list[PatternGroup]:
    groups: list[PatternGroup] = []

    for raw_group in raw_patterns:
        if isinstance(raw_group, str):
            raw_items = [raw_group]
        else:
            try:
                raw_items = list(raw_group)
            except TypeError:
                raw_items = [raw_group]

        normalized_group: PatternGroup = []
        for raw_item in raw_items:
            if isinstance(raw_item, str):
                raw_alternatives = [raw_item]
            else:
                try:
                    raw_alternatives = list(raw_item)
                except TypeError:
                    raw_alternatives = [raw_item]

            normalized_slot = list(
                dict.fromkeys(
                    normalize_text(str(alternative).strip())
                    for alternative in raw_alternatives
                    if normalize_text(str(alternative).strip())
                )
            )
            if normalized_slot:
                normalized_group.append(normalized_slot)

        if normalized_group:
            groups.append(normalized_group)

    return groups


def matches_pattern_group(normalized_answer: str, pattern_group: PatternGroup) -> bool:
    return all(
        any(alternative in normalized_answer for alternative in slot)
        for slot in pattern_group
    )


def answer_matches_patterns(
    answer: str,
    answer_patterns: list[PatternGroup],
) -> bool:
    normalized_answer = normalize_text(answer)
    return any(
        matches_pattern_group(normalized_answer, pattern_group)
        for pattern_group in answer_patterns
    )


def count_pattern_group_hits(
    normalized_text: str,
    pattern_group: PatternGroup,
) -> int:
    return sum(
        1
        for slot in pattern_group
        if any(alternative in normalized_text for alternative in slot)
    )

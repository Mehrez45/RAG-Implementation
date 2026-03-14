import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunking import Chunk
from src.retrieval.storage import load_index
from src.utilities.answer_patterns import (
    count_pattern_group_hits,
    normalize_text,
    parse_answer_patterns,
)
MAX_WINDOW_SIZE = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Refresh benchmark relevant_chunk_ids against the current index using "
            "relevant_pages plus answer-pattern matching."
        )
    )
    parser.add_argument(
        "--benchmark",
        default="experiments/eval_queries/single_hop.json",
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to overwriting the benchmark file.",
    )
    return parser.parse_args()


def select_page_chunk_ids(
    chunks: list[Chunk],
    answer_patterns: list[object],
    fallback_chunk_ids: list[str],
) -> list[str]:
    if not chunks:
        return []

    normalized_chunks = [
        (chunk, normalize_text(chunk.text))
        for chunk in sorted(chunks, key=lambda item: item.start_token)
    ]
    normalized_groups = parse_answer_patterns(answer_patterns)

    best_window: Optional[list[Chunk]] = None
    best_score: Optional[tuple[int, float, int, int]] = None

    for group in normalized_groups:
        group_size = len(group)
        max_window_size = min(MAX_WINDOW_SIZE, len(normalized_chunks))
        for window_size in range(1, max_window_size + 1):
            for start in range(len(normalized_chunks) - window_size + 1):
                window = normalized_chunks[start : start + window_size]
                window_text = " ".join(text for _, text in window)
                hits = count_pattern_group_hits(window_text, group)
                if hits == 0:
                    continue

                score = (
                    1 if hits == group_size else 0,
                    hits / group_size,
                    -window_size,
                    -start,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_window = [chunk for chunk, _ in window]

    if best_window is not None:
        return [chunk.chunk_id for chunk in best_window]

    fallback_on_page = {chunk.chunk_id for chunk, _ in normalized_chunks}
    preserved = [chunk_id for chunk_id in fallback_chunk_ids if chunk_id in fallback_on_page]
    if preserved:
        return preserved

    return [normalized_chunks[0][0].chunk_id]


def main():
    args = parse_args()
    benchmark_path = Path(args.benchmark)
    output_path = Path(args.output) if args.output else benchmark_path
    payload = json.loads(benchmark_path.read_text())

    _, chunks = load_index()
    chunks_by_page: dict[tuple[str, int], list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_page[(chunk.doc_id, chunk.page_number)].append(chunk)

    changed_queries = 0
    total_chunk_labels_before = 0
    total_chunk_labels_after = 0

    for query in payload.get("queries", []):
        original_chunk_ids = list(query.get("relevant_chunk_ids", []))
        total_chunk_labels_before += len(original_chunk_ids)

        refreshed_chunk_ids: list[str] = []
        for page in query.get("relevant_pages", []):
            doc_id = page.get("doc_id")
            page_number = page.get("page_number")
            if doc_id is None or page_number is None:
                continue

            page_chunks = chunks_by_page.get((doc_id, page_number), [])
            refreshed_chunk_ids.extend(
                select_page_chunk_ids(
                    chunks=page_chunks,
                    answer_patterns=query.get("answer_patterns", []),
                    fallback_chunk_ids=original_chunk_ids,
                )
            )

        # Deduplicate while preserving page order.
        deduped_chunk_ids = list(dict.fromkeys(refreshed_chunk_ids))
        query["relevant_chunk_ids"] = deduped_chunk_ids
        total_chunk_labels_after += len(deduped_chunk_ids)

        if deduped_chunk_ids != original_chunk_ids:
            changed_queries += 1

    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Updated chunk labels for {changed_queries} queries.")
    print(
        f"Total chunk labels: {total_chunk_labels_before} -> {total_chunk_labels_after}"
    )
    print(f"Wrote refreshed benchmark to {output_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for entry in (PROJECT_ROOT, EXPERIMENTS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from retrieval_eval import (
    build_ranked_ids,
    compute_metric_bundle,
    load_benchmark,
    print_metric_bundle,
)
from src.pipeline.retrieval_context import prepare_ranking
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import FaissRetriever
from src.retrieval.storage import load_index

DEFAULT_BENCHMARK_PATH = "experiments/eval_queries/single_hop.json"
DEFAULT_PROFILES = ("faiss", "rerank")
DEFAULT_CANDIDATE_K_VALUES = (24, 40, 60, 80)
DEFAULT_THRESHOLDS = (0.15, 0.25, 0.35, 0.45)
DEFAULT_K_VALUES = (1, 3, 5, 10)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievalTuningResult:
    profile: str
    candidate_k: int
    threshold: float
    avg_latency_ms: float
    chunk_metrics: Optional[dict]
    page_metrics: Optional[dict]
    doc_metrics: Optional[dict]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep candidate_k and retrieval threshold settings on the current "
            "indexed corpus to find stronger benchmark configurations."
        )
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--profiles",
        default="faiss,rerank",
        help="Comma-separated profiles to evaluate: faiss, rerank.",
    )
    parser.add_argument(
        "--candidate-k-values",
        default="24,40,60,80",
        help="Comma-separated list of candidate_k values to test.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.15,0.25,0.35,0.45",
        help="Comma-separated list of FAISS similarity thresholds to test.",
    )
    parser.add_argument(
        "--k-values",
        default="1,3,5,10",
        help="Comma-separated cutoffs for precision@k, recall@k, and hit-rate@k.",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder model used when the rerank profile is enabled.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the full sweep summary as JSON.",
    )
    return parser.parse_args()


def parse_int_list(raw_value: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw_value.split(",") if part.strip()})
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_float_list(raw_value: str) -> list[float]:
    values = sorted(
        {float(part.strip()) for part in raw_value.split(",") if part.strip()}
    )
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def parse_profiles(raw_value: str) -> list[str]:
    requested = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not requested:
        raise ValueError("Provide at least one profile.")

    unknown = [name for name in requested if name not in DEFAULT_PROFILES]
    if unknown:
        raise ValueError(
            f"Unknown profile(s): {', '.join(unknown)}. "
            "Available: faiss, rerank."
        )

    return requested


def print_leaderboard(results: list[RetrievalTuningResult]) -> None:
    print("\n=== Leaderboard (sorted by page-level MRR, then page-level P@1) ===")
    ranked = sorted(
        results,
        key=lambda item: (
            item.page_metrics["mrr"] if item.page_metrics is not None else -1.0,
            item.page_metrics["precision_at_k"].get("p@1", -1.0)
            if item.page_metrics is not None
            else -1.0,
        ),
        reverse=True,
    )

    for index, result in enumerate(ranked, start=1):
        page_mrr = (
            "n/a"
            if result.page_metrics is None
            else f"{result.page_metrics['mrr']:.3f}"
        )
        page_p1 = (
            "n/a"
            if result.page_metrics is None
            else f"{result.page_metrics['precision_at_k']['p@1']:.3f}"
        )
        page_r5 = (
            "n/a"
            if result.page_metrics is None
            else f"{result.page_metrics['recall_at_k'].get('r@5', 0.0):.3f}"
        )
        print(
            f"{index}. profile={result.profile}"
            f" | candidate_k={result.candidate_k}"
            f" | threshold={result.threshold:.2f}"
            f" | page_mrr={page_mrr}"
            f" | page_p@1={page_p1}"
            f" | page_r@5={page_r5}"
            f" | latency_ms={result.avg_latency_ms:.2f}"
        )


def main():
    args = parse_args()
    profiles = parse_profiles(args.profiles)
    candidate_k_values = parse_int_list(args.candidate_k_values)
    thresholds = parse_float_list(args.thresholds)
    k_values = parse_int_list(args.k_values)

    benchmark_queries = load_benchmark(args.benchmark)
    if not benchmark_queries:
        raise ValueError("Benchmark file has no queries.")

    index, chunks = load_index()
    retriever = FaissRetriever(index, chunks)

    reranker = None
    if "rerank" in profiles:
        reranker = CrossEncoderReranker(args.reranker_model)

    sweep_results: list[RetrievalTuningResult] = []

    for profile in profiles:
        active_reranker = reranker if profile == "rerank" else None

        for candidate_k in candidate_k_values:
            effective_candidate_k = max(candidate_k, max(k_values))
            for threshold in thresholds:
                print(
                    f"\n=== Config: profile={profile}, candidate_k={effective_candidate_k}, "
                    f"threshold={threshold:.2f} ==="
                )

                chunk_rankings: list[list[str]] = []
                page_rankings: list[list[str]] = []
                doc_rankings: list[list[str]] = []
                chunk_labels: list[set[str]] = []
                page_labels: list[set[str]] = []
                doc_labels: list[set[str]] = []
                latencies_ms: list[float] = []

                for benchmark_query in benchmark_queries:
                    started_at = time.perf_counter()
                    ranking = prepare_ranking(
                        query=benchmark_query.query,
                        retriever=retriever,
                        candidate_k=effective_candidate_k,
                        threshold=threshold,
                        reranker=active_reranker,
                    )
                    latencies_ms.append((time.perf_counter() - started_at) * 1000.0)

                    chunk_rankings.append(
                        build_ranked_ids(ranking.ranked_chunks, "chunk")
                    )
                    page_rankings.append(
                        build_ranked_ids(ranking.ranked_chunks, "page")
                    )
                    doc_rankings.append(build_ranked_ids(ranking.ranked_chunks, "doc"))
                    chunk_labels.append(benchmark_query.relevant_chunk_ids)
                    page_labels.append(benchmark_query.relevant_pages)
                    doc_labels.append(benchmark_query.relevant_doc_ids)

                chunk_metrics = compute_metric_bundle(
                    chunk_rankings,
                    chunk_labels,
                    k_values,
                )
                page_metrics = compute_metric_bundle(
                    page_rankings,
                    page_labels,
                    k_values,
                )
                doc_metrics = compute_metric_bundle(
                    doc_rankings,
                    doc_labels,
                    k_values,
                )

                avg_latency_ms = statistics.mean(latencies_ms)
                print(f"avg_latency_ms={avg_latency_ms:.2f}")
                print_metric_bundle("Chunk-level", chunk_metrics)
                print_metric_bundle("Page-level", page_metrics)
                print_metric_bundle("Doc-level", doc_metrics)

                sweep_results.append(
                    RetrievalTuningResult(
                        profile=profile,
                        candidate_k=effective_candidate_k,
                        threshold=threshold,
                        avg_latency_ms=avg_latency_ms,
                        chunk_metrics=asdict(chunk_metrics)
                        if chunk_metrics is not None
                        else None,
                        page_metrics=asdict(page_metrics)
                        if page_metrics is not None
                        else None,
                        doc_metrics=asdict(doc_metrics)
                        if doc_metrics is not None
                        else None,
                    )
                )

    print_leaderboard(sweep_results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": args.benchmark,
            "profiles": profiles,
            "candidate_k_values": candidate_k_values,
            "thresholds": thresholds,
            "results": [asdict(result) for result in sweep_results],
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote sweep summary to {output_path}")


if __name__ == "__main__":
    main()

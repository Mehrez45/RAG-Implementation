import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for entry in (PROJECT_ROOT, EXPERIMENTS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from retrieval_eval import BenchmarkQuery, load_benchmark
from src.app.runners import RunResult, build_runner
from src.app.runtime import (
    AppRuntime,
    DEFAULT_AGENTIC_MAX_REVISIONS,
    DEFAULT_AGENTIC_REVIEWER_ACCEPT_SCORE,
    DEFAULT_CANDIDATE_K,
    DEFAULT_FINAL_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_THRESHOLD,
)
from src.llm.local_llm import LocalLLM
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import FaissRetriever
from src.retrieval.storage import load_index
from src.utilities.answer_patterns import answer_matches_patterns, normalize_text
from src.utilities.document_aliases import DocumentAliasCatalog, build_document_alias_catalog


DEFAULT_BENCHMARK_PATH = "experiments/eval_queries/single_hop.json"
DEFAULT_PROFILES = "vanilla_no_rerank,agentic_rerank"
@dataclass(frozen=True)
class ProfileSpec:
    name: str
    mode: str
    use_reranker: bool


@dataclass
class QueryResult:
    query_id: str
    query: str
    matched: Optional[bool]
    answer_labeled: bool
    abstained: bool
    latency_ms: float
    route: str
    route_reason: str
    revision_count: int
    failure_reason: str
    answer: str


@dataclass
class ProfileSummary:
    profile: str
    mode: str
    reranker_enabled: bool
    total_queries: int
    labeled_queries: int
    matched_queries: int
    mismatch_queries: int
    answer_match_rate: Optional[float]
    abstain_rate: float
    false_abstain_rate: Optional[float]
    avg_latency_ms: float
    avg_revision_count: float
    route_distribution: dict[str, int]
    route_reason_distribution: dict[str, int]
    results: list[QueryResult]


PROFILE_REGISTRY = {
    "vanilla_no_rerank": ProfileSpec(
        name="vanilla_no_rerank",
        mode="vanilla",
        use_reranker=False,
    ),
    "vanilla_rerank": ProfileSpec(
        name="vanilla_rerank",
        mode="vanilla",
        use_reranker=True,
    ),
    "agentic_no_rerank": ProfileSpec(
        name="agentic_no_rerank",
        mode="agentic",
        use_reranker=False,
    ),
    "agentic_rerank": ProfileSpec(
        name="agentic_rerank",
        mode="agentic",
        use_reranker=True,
    ),
}

ABSTAIN_MARKERS = (
    "i dont know",
    "i do not know",
    "not enough context",
    "not enough information",
    "cannot answer",
    "cant answer",
    "unable to answer",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare end-to-end answer quality and latency across vanilla and "
            "agentic RAG profiles."
        )
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--profiles",
        default=DEFAULT_PROFILES,
        help=(
            "Comma-separated profile names. "
            "Available: vanilla_no_rerank, vanilla_rerank, "
            "agentic_no_rerank, agentic_rerank."
        ),
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Number of retrieval candidates kept before final context selection.",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=DEFAULT_FINAL_K,
        help="Number of chunks kept for the final prompt context.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Similarity threshold applied to FAISS retrieval results.",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder model used when a profile enables reranking.",
    )
    parser.add_argument(
        "--use-expander",
        action="store_true",
        help="Enable query expansion for every profile in the comparison.",
    )
    parser.add_argument(
        "--use-decomposer",
        action="store_true",
        help="Enable query decomposition for every profile in the comparison.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally evaluate only the first N queries for a quick smoke test.",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=0,
        help="Print up to N mismatched labeled answers per profile.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=DEFAULT_AGENTIC_MAX_REVISIONS,
        help="Maximum number of extractor passes allowed in agentic mode.",
    )
    parser.add_argument(
        "--reviewer-accept-score",
        type=float,
        default=DEFAULT_AGENTIC_REVIEWER_ACCEPT_SCORE,
        help="Skip the LLM reviewer when the top rerank score exceeds this value.",
    )
    parser.add_argument(
        "--force-retrieval",
        action="store_true",
        help=(
            "Force agentic profiles to retrieve for every query. Useful for "
            "document-grounded benchmark ablations without changing normal app behavior."
        ),
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the full comparison summary as JSON.",
    )
    return parser.parse_args()


def is_abstained(answer: str) -> bool:
    normalized_answer = normalize_text(answer)
    return any(marker in normalized_answer for marker in ABSTAIN_MARKERS)


def answer_matches(answer: str, benchmark_query: BenchmarkQuery) -> Optional[bool]:
    if not benchmark_query.answer_patterns:
        return None

    return answer_matches_patterns(answer, benchmark_query.answer_patterns)


def parse_profiles(raw_value: str) -> list[ProfileSpec]:
    requested = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not requested:
        raise ValueError("At least one profile must be requested.")

    unknown = [name for name in requested if name not in PROFILE_REGISTRY]
    if unknown:
        known = ", ".join(sorted(PROFILE_REGISTRY))
        raise ValueError(
            f"Unknown profile(s): {', '.join(unknown)}. Available profiles: {known}"
        )

    return [PROFILE_REGISTRY[name] for name in requested]


def build_runtime(
    llm: LocalLLM,
    retriever: FaissRetriever,
    candidate_k: int,
    final_k: int,
    threshold: float,
    expander: Optional[QueryExpander],
    decomposer: Optional[QueryDecomposer],
    reranker: Optional[CrossEncoderReranker],
    max_revisions: int,
    reviewer_accept_score: float,
    force_retrieval: bool,
    document_aliases: DocumentAliasCatalog,
) -> AppRuntime:
    return AppRuntime(
        llm=llm,
        retriever=retriever,
        expander=expander,
        decomposer=decomposer,
        reranker=reranker,
        candidate_k=candidate_k,
        final_k=final_k,
        threshold=threshold,
        max_revisions=max_revisions,
        reviewer_accept_score=reviewer_accept_score,
        force_retrieval=force_retrieval,
        document_aliases=document_aliases,
    )


def evaluate_profile(
    profile: ProfileSpec,
    runner,
    benchmark_queries: list[BenchmarkQuery],
) -> ProfileSummary:
    results: list[QueryResult] = []
    route_counts: Counter[str] = Counter()
    route_reason_counts: Counter[str] = Counter()

    for benchmark_query in benchmark_queries:
        started_at = time.perf_counter()
        run_result: RunResult = runner.run(benchmark_query.query)
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        matched = answer_matches(run_result.answer, benchmark_query)
        abstained = is_abstained(run_result.answer)
        route_counts[run_result.route or "unknown"] += 1
        route_reason_counts[run_result.route_reason or "unknown"] += 1

        results.append(
            QueryResult(
                query_id=benchmark_query.query_id,
                query=benchmark_query.query,
                matched=matched,
                answer_labeled=matched is not None,
                abstained=abstained,
                latency_ms=latency_ms,
                route=run_result.route or "unknown",
                route_reason=run_result.route_reason or "unknown",
                revision_count=run_result.revision_count,
                failure_reason=run_result.failure_reason,
                answer=run_result.answer,
            )
        )

    labeled_results = [result for result in results if result.answer_labeled]
    matched_queries = sum(1 for result in labeled_results if result.matched)
    mismatch_queries = sum(1 for result in labeled_results if result.matched is False)
    false_abstains = sum(
        1 for result in labeled_results if result.abstained and result.matched is False
    )

    return ProfileSummary(
        profile=profile.name,
        mode=profile.mode,
        reranker_enabled=profile.use_reranker,
        total_queries=len(results),
        labeled_queries=len(labeled_results),
        matched_queries=matched_queries,
        mismatch_queries=mismatch_queries,
        answer_match_rate=(
            matched_queries / len(labeled_results) if labeled_results else None
        ),
        abstain_rate=statistics.mean(
            1.0 if result.abstained else 0.0 for result in results
        ),
        false_abstain_rate=(
            false_abstains / len(labeled_results) if labeled_results else None
        ),
        avg_latency_ms=statistics.mean(result.latency_ms for result in results),
        avg_revision_count=statistics.mean(
            result.revision_count for result in results
        ),
        route_distribution=dict(route_counts),
        route_reason_distribution=dict(route_reason_counts),
        results=results,
    )


def print_summary(summary: ProfileSummary) -> None:
    print(
        f"\n=== {summary.profile} "
        f"(mode={summary.mode}, reranker={'on' if summary.reranker_enabled else 'off'}) ==="
    )
    if summary.answer_match_rate is None:
        print("answer_match_rate=n/a (no answer_patterns in benchmark)")
    else:
        print(
            f"answer_match_rate={summary.answer_match_rate:.3f}"
            f" ({summary.matched_queries}/{summary.labeled_queries})"
        )
        print(
            f"false_abstain_rate={summary.false_abstain_rate:.3f}"
            f" | mismatches={summary.mismatch_queries}"
        )

    route_breakdown = ", ".join(
        f"{route}:{count}" for route, count in sorted(summary.route_distribution.items())
    )
    print(
        f"avg_latency_ms={summary.avg_latency_ms:.2f}"
        f" | avg_revisions={summary.avg_revision_count:.2f}"
        f" | abstain_rate={summary.abstain_rate:.3f}"
    )
    print(f"routes={route_breakdown or 'n/a'}")


def print_deltas(summaries: list[ProfileSummary]) -> None:
    if len(summaries) < 2:
        return

    baseline = summaries[0]
    print(f"\n=== Delta vs {baseline.profile} ===")
    for summary in summaries[1:]:
        latency_delta = summary.avg_latency_ms - baseline.avg_latency_ms
        revision_delta = summary.avg_revision_count - baseline.avg_revision_count
        if (
            baseline.answer_match_rate is not None
            and summary.answer_match_rate is not None
        ):
            match_delta = summary.answer_match_rate - baseline.answer_match_rate
            print(
                f"{summary.profile}"
                f" | answer_match_delta={match_delta:+.3f}"
                f" | latency_delta_ms={latency_delta:+.2f}"
                f" | revision_delta={revision_delta:+.2f}"
            )
        else:
            print(
                f"{summary.profile}"
                f" | latency_delta_ms={latency_delta:+.2f}"
                f" | revision_delta={revision_delta:+.2f}"
            )


def print_failures(summary: ProfileSummary, limit: int) -> None:
    if limit <= 0:
        return

    failures = [
        result
        for result in summary.results
        if result.answer_labeled and result.matched is False
    ][:limit]
    if not failures:
        return

    print(f"\n--- Sample mismatches for {summary.profile} ---")
    for result in failures:
        answer_preview = " ".join(result.answer.split())
        print(
            f"{result.query_id}"
            f" | route={result.route}"
            f" | revisions={result.revision_count}"
            f" | latency_ms={result.latency_ms:.2f}"
            f" | abstained={result.abstained}"
        )
        print(f"Q: {result.query}")
        print(f"A: {answer_preview[:240]}")


def main():
    args = parse_args()
    profiles = parse_profiles(args.profiles)
    benchmark_queries = load_benchmark(args.benchmark)
    if not benchmark_queries:
        raise ValueError("Benchmark file has no queries.")

    if args.limit > 0:
        benchmark_queries = benchmark_queries[: args.limit]

    index, chunks = load_index()
    retriever = FaissRetriever(index, chunks)
    document_aliases = build_document_alias_catalog(chunks)
    llm = LocalLLM()

    expander = QueryExpander(llm=llm) if args.use_expander else None
    decomposer = QueryDecomposer(llm=llm) if args.use_decomposer else None
    shared_reranker = None
    if any(profile.use_reranker for profile in profiles):
        shared_reranker = CrossEncoderReranker(args.reranker_model)

    summaries: list[ProfileSummary] = []
    for profile in profiles:
        runtime = build_runtime(
            llm=llm,
            retriever=retriever,
            candidate_k=max(args.final_k, args.candidate_k),
            final_k=args.final_k,
            threshold=args.threshold,
            expander=expander,
            decomposer=decomposer,
            reranker=shared_reranker if profile.use_reranker else None,
            max_revisions=max(1, args.max_revisions),
            reviewer_accept_score=args.reviewer_accept_score,
            force_retrieval=args.force_retrieval,
            document_aliases=document_aliases,
        )
        runner = build_runner(profile.mode, runtime)
        summary = evaluate_profile(profile, runner, benchmark_queries)
        summaries.append(summary)
        print_summary(summary)
        print_failures(summary, args.show_failures)

    print_deltas(summaries)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": args.benchmark,
            "profiles": [summary.profile for summary in summaries],
            "candidate_k": max(args.final_k, args.candidate_k),
            "final_k": args.final_k,
            "threshold": args.threshold,
            "use_expander": args.use_expander,
            "use_decomposer": args.use_decomposer,
            "max_revisions": max(1, args.max_revisions),
            "reviewer_accept_score": args.reviewer_accept_score,
            "force_retrieval": args.force_retrieval,
            "summaries": [asdict(summary) for summary in summaries],
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved comparison summary to {output_path}")


if __name__ == "__main__":
    main()

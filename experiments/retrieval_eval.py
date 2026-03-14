import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.retrieval_context import prepare_ranking
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import CrossEncoderReranker, RankedChunk
from src.retrieval.retriever import FaissRetriever
from src.retrieval.storage import load_index
from src.utilities.answer_patterns import PatternGroup, parse_answer_patterns


DEFAULT_BENCHMARK_PATH = "experiments/eval_queries/single_hop.json"
DEFAULT_K_VALUES = [1, 3, 5, 10]
DEFAULT_CANDIDATE_K = 24
DEFAULT_THRESHOLD = 0.35
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class BenchmarkQuery:
    query_id: str
    query: str
    relevant_chunk_ids: set[str]
    relevant_doc_ids: set[str]
    relevant_pages: set[str]
    answer_patterns: list[PatternGroup]
    notes: str = ""


@dataclass
class MetricBundle:
    precision_at_k: dict[str, float]
    recall_at_k: dict[str, float]
    hit_rate_at_k: dict[str, float]
    mrr: float
    evaluated_queries: int


@dataclass
class ProfileSummary:
    profile: str
    candidate_k: int
    threshold: float
    avg_latency_ms: float
    chunk: Optional[MetricBundle]
    page: Optional[MetricBundle]
    doc: Optional[MetricBundle]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval precision/recall on a labeled benchmark."
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--k-values",
        default="1,3,5,10",
        help="Comma-separated cutoffs for precision@k, recall@k, and hit-rate@k.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Number of deduplicated candidates to keep before reranking.",
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
        help="Cross-encoder model name used for reranking.",
    )
    parser.add_argument(
        "--use-expander",
        action="store_true",
        help="Enable query expansion during evaluation.",
    )
    parser.add_argument(
        "--use-decomposer",
        action="store_true",
        help="Enable query decomposition during evaluation.",
    )
    parser.add_argument(
        "--show-top-k",
        type=int,
        default=0,
        help="Print the top retrieved chunks for each query and profile.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the benchmark summary as JSON.",
    )
    return parser.parse_args()


def load_benchmark(path: str) -> list[BenchmarkQuery]:
    payload = json.loads(Path(path).read_text())
    queries = payload.get("queries", [])
    benchmark_queries: list[BenchmarkQuery] = []

    for index, item in enumerate(queries, start=1):
        query_id = item.get("id") or f"query_{index}"
        relevant_pages = {
            f"{page['doc_id']}::{page['page_number']}"
            for page in item.get("relevant_pages", [])
            if "doc_id" in page and "page_number" in page
        }

        answer_patterns = parse_answer_patterns(item.get("answer_patterns", []))

        benchmark_queries.append(
            BenchmarkQuery(
                query_id=query_id,
                query=item["query"],
                relevant_chunk_ids=set(item.get("relevant_chunk_ids", [])),
                relevant_doc_ids=set(item.get("relevant_doc_ids", [])),
                relevant_pages=relevant_pages,
                answer_patterns=answer_patterns,
                notes=item.get("notes", ""),
            )
        )

    return benchmark_queries


def validate_benchmark(queries: list[BenchmarkQuery], retriever: FaissRetriever) -> None:
    chunk_ids = {chunk.chunk_id for chunk in retriever.chunks}
    doc_ids = {chunk.doc_id for chunk in retriever.chunks}
    page_ids = {f"{chunk.doc_id}::{chunk.page_number}" for chunk in retriever.chunks}

    for benchmark_query in queries:
        unknown_chunks = sorted(benchmark_query.relevant_chunk_ids - chunk_ids)
        unknown_docs = sorted(benchmark_query.relevant_doc_ids - doc_ids)
        unknown_pages = sorted(benchmark_query.relevant_pages - page_ids)

        if unknown_chunks:
            print(
                f"[warn] {benchmark_query.query_id} has unknown chunk ids: "
                + ", ".join(unknown_chunks)
            )
        if unknown_docs:
            print(
                f"[warn] {benchmark_query.query_id} has unknown doc ids: "
                + ", ".join(unknown_docs)
            )
        if unknown_pages:
            print(
                f"[warn] {benchmark_query.query_id} has unknown page labels: "
                + ", ".join(unknown_pages)
            )


def build_ranked_ids(
    ranked_chunks: list[RankedChunk],
    label_type: str,
) -> list[str]:
    if label_type == "chunk":
        return [item.chunk.chunk_id for item in ranked_chunks]

    ordered: list[str] = []
    seen: set[str] = set()
    for item in ranked_chunks:
        if label_type == "page":
            value = f"{item.chunk.doc_id}::{item.chunk.page_number}"
        else:
            value = item.chunk.doc_id

        if value not in seen:
            seen.add(value)
            ordered.append(value)

    return ordered


def compute_metric_bundle(
    rankings: list[list[str]],
    relevant_sets: list[set[str]],
    k_values: list[int],
) -> Optional[MetricBundle]:
    filtered = [
        (ranking, relevant)
        for ranking, relevant in zip(rankings, relevant_sets)
        if relevant
    ]
    if not filtered:
        return None

    precision_values = {f"p@{k}": [] for k in k_values}
    recall_values = {f"r@{k}": [] for k in k_values}
    hit_rate_values = {f"hit@{k}": [] for k in k_values}
    reciprocal_ranks = []

    for ranking, relevant in filtered:
        for k in k_values:
            top_k = ranking[:k]
            hits = sum(1 for item in top_k if item in relevant)
            precision_values[f"p@{k}"].append(hits / k)
            recall_values[f"r@{k}"].append(hits / len(relevant))
            hit_rate_values[f"hit@{k}"].append(1.0 if hits > 0 else 0.0)

        reciprocal_rank = 0.0
        for rank, item in enumerate(ranking, start=1):
            if item in relevant:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

    return MetricBundle(
        precision_at_k={
            metric: statistics.mean(values) for metric, values in precision_values.items()
        },
        recall_at_k={
            metric: statistics.mean(values) for metric, values in recall_values.items()
        },
        hit_rate_at_k={
            metric: statistics.mean(values) for metric, values in hit_rate_values.items()
        },
        mrr=statistics.mean(reciprocal_ranks),
        evaluated_queries=len(filtered),
    )


def print_top_results(
    profile_name: str,
    benchmark_query: BenchmarkQuery,
    ranked_chunks: list[RankedChunk],
    limit: int,
) -> None:
    print(f"\n[{profile_name}] {benchmark_query.query_id}: {benchmark_query.query}")
    for index, item in enumerate(ranked_chunks[:limit], start=1):
        snippet = " ".join(item.chunk.text.split())[:140]
        rerank_score = (
            "n/a" if item.rerank_score is None else f"{item.rerank_score:.4f}"
        )
        print(
            f"  {index}. {item.chunk.chunk_id}"
            f" | doc={item.chunk.doc_id}"
            f" | page={item.chunk.page_number}"
            f" | retrieval={item.retrieval_score:.4f}"
            f" | rerank={rerank_score}"
            f" | {snippet}"
        )


def print_metric_bundle(title: str, metrics: Optional[MetricBundle]) -> None:
    if metrics is None:
        print(f"{title}: no labels provided")
        return

    print(f"{title}:")
    precision = ", ".join(
        f"{name}={value:.3f}" for name, value in metrics.precision_at_k.items()
    )
    recall = ", ".join(
        f"{name}={value:.3f}" for name, value in metrics.recall_at_k.items()
    )
    hit_rate = ", ".join(
        f"{name}={value:.3f}" for name, value in metrics.hit_rate_at_k.items()
    )
    print(f"  evaluated_queries={metrics.evaluated_queries}")
    print(f"  precision: {precision}")
    print(f"  recall: {recall}")
    print(f"  hit-rate: {hit_rate}")
    print(f"  mrr={metrics.mrr:.3f}")


def print_delta(title: str, baseline: Optional[MetricBundle], candidate: Optional[MetricBundle]) -> None:
    if baseline is None or candidate is None:
        return

    print(f"{title}:")
    for metric_name, value in candidate.precision_at_k.items():
        print(
            f"  {metric_name}: {value - baseline.precision_at_k[metric_name]:+.3f}"
        )
    for metric_name, value in candidate.recall_at_k.items():
        print(
            f"  {metric_name}: {value - baseline.recall_at_k[metric_name]:+.3f}"
        )
    for metric_name, value in candidate.hit_rate_at_k.items():
        print(
            f"  {metric_name}: {value - baseline.hit_rate_at_k[metric_name]:+.3f}"
        )
    print(f"  mrr: {candidate.mrr - baseline.mrr:+.3f}")


def main():
    args = parse_args()
    k_values = sorted({int(value) for value in args.k_values.split(",") if value.strip()})
    if not k_values:
        raise ValueError("Provide at least one k value.")

    queries = load_benchmark(args.benchmark)
    if not queries:
        raise ValueError(
            "Benchmark file has no queries. Add labeled queries before running evaluation."
        )

    index, chunks = load_index()
    retriever = FaissRetriever(index, chunks)
    validate_benchmark(queries, retriever)

    llm = None
    expander = None
    decomposer = None
    if args.use_expander or args.use_decomposer:
        from src.llm.local_llm import LocalLLM

        llm = LocalLLM()
        if args.use_expander:
            expander = QueryExpander(llm=llm)
        if args.use_decomposer:
            decomposer = QueryDecomposer(llm=llm)

    profiles = {
        "faiss": None,
        "rerank": CrossEncoderReranker(args.reranker_model),
    }

    summary_payload = {
        "benchmark": str(Path(args.benchmark)),
        "k_values": k_values,
        "candidate_k": args.candidate_k,
        "threshold": args.threshold,
        "profiles": [],
    }

    baseline_summary: Optional[ProfileSummary] = None

    for profile_name, reranker in profiles.items():
        chunk_rankings: list[list[str]] = []
        page_rankings: list[list[str]] = []
        doc_rankings: list[list[str]] = []
        chunk_labels: list[set[str]] = []
        page_labels: list[set[str]] = []
        doc_labels: list[set[str]] = []
        latencies_ms: list[float] = []

        print(f"\n=== Profile: {profile_name} ===")

        effective_candidate_k = max(args.candidate_k, max(k_values))
        for benchmark_query in queries:
            started_at = time.perf_counter()
            ranking = prepare_ranking(
                query=benchmark_query.query,
                retriever=retriever,
                candidate_k=effective_candidate_k,
                threshold=args.threshold,
                decomposer=decomposer,
                expander=expander,
                reranker=reranker,
            )
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            latencies_ms.append(elapsed_ms)

            chunk_rankings.append(build_ranked_ids(ranking.ranked_chunks, "chunk"))
            page_rankings.append(build_ranked_ids(ranking.ranked_chunks, "page"))
            doc_rankings.append(build_ranked_ids(ranking.ranked_chunks, "doc"))
            chunk_labels.append(benchmark_query.relevant_chunk_ids)
            page_labels.append(benchmark_query.relevant_pages)
            doc_labels.append(benchmark_query.relevant_doc_ids)

            if args.show_top_k > 0:
                print_top_results(
                    profile_name=profile_name,
                    benchmark_query=benchmark_query,
                    ranked_chunks=ranking.ranked_chunks,
                    limit=args.show_top_k,
                )

        summary = ProfileSummary(
            profile=profile_name,
            candidate_k=effective_candidate_k,
            threshold=args.threshold,
            avg_latency_ms=statistics.mean(latencies_ms),
            chunk=compute_metric_bundle(chunk_rankings, chunk_labels, k_values),
            page=compute_metric_bundle(page_rankings, page_labels, k_values),
            doc=compute_metric_bundle(doc_rankings, doc_labels, k_values),
        )

        print(f"avg_latency_ms={summary.avg_latency_ms:.2f}")
        print_metric_bundle("Chunk-level", summary.chunk)
        print_metric_bundle("Page-level", summary.page)
        print_metric_bundle("Doc-level", summary.doc)

        summary_payload["profiles"].append(asdict(summary))
        if baseline_summary is None:
            baseline_summary = summary
        else:
            print("\nDelta vs faiss:")
            print_delta("Chunk-level", baseline_summary.chunk, summary.chunk)
            print_delta("Page-level", baseline_summary.page, summary.page)
            print_delta("Doc-level", baseline_summary.doc, summary.doc)
            print(f"Latency delta: {summary.avg_latency_ms - baseline_summary.avg_latency_ms:+.2f} ms")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary_payload, indent=2))
        print(f"\nWrote summary to {output_path}")


if __name__ == "__main__":
    main()

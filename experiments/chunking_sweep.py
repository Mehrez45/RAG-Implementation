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
from src.ingestion.chunking import analytics, chunk_docs
from src.ingestion.pdf_loader import load_pdfs
from src.pipeline.retrieval_context import prepare_ranking
from src.retrieval.embeddings import embed_chunks
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import FaissRetriever
from src.retrieval.storage import build_faiss_index

DEFAULT_BENCHMARK_PATH = "experiments/eval_queries/single_hop.json"
DEFAULT_CHUNK_SIZES = [256, 384, 512, 768]
DEFAULT_OVERLAPS = [0, 32, 64, 96]
DEFAULT_K_VALUES = [1, 3, 5, 10]
DEFAULT_CANDIDATE_K = 24
DEFAULT_THRESHOLD = 0.35
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class ChunkingSweepResult:
    profile: str
    chunk_size: int
    overlap: int
    num_chunks: int
    avg_tokens: float
    median_tokens: float
    p95_tokens: float
    avg_latency_ms: float
    page_metrics: Optional[dict]
    doc_metrics: Optional[dict]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate chunk-size and overlap settings on the labeled retrieval benchmark."
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/raw/pdfs",
        help="Directory containing the source PDFs.",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="256,384,512,768",
        help="Comma-separated list of chunk sizes to test.",
    )
    parser.add_argument(
        "--overlaps",
        default="0,32,64,96",
        help="Comma-separated list of chunk overlaps to test.",
    )
    parser.add_argument(
        "--profile",
        choices=("faiss", "rerank"),
        default="rerank",
        help="Retrieval profile to benchmark for each chunking configuration.",
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
        help="Number of candidates to keep before reranking or final ranking.",
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
        help="Cross-encoder model used when profile=rerank.",
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
        "--output-json",
        help="Optional path to write the sweep summary as JSON.",
    )
    return parser.parse_args()


def parse_int_list(raw_value: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw_value.split(",") if part.strip()})
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def metric_to_dict(metric_bundle: Optional[object]) -> Optional[dict]:
    if metric_bundle is None:
        return None
    return asdict(metric_bundle)


def print_leaderboard(results: list[ChunkingSweepResult]) -> None:
    print("\n=== Leaderboard (sorted by page-level MRR) ===")
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
            f"{index}. size={result.chunk_size} overlap={result.overlap}"
            f" | chunks={result.num_chunks}"
            f" | page_mrr={page_mrr}"
            f" | page_p@1={page_p1}"
            f" | page_r@5={page_r5}"
            f" | latency_ms={result.avg_latency_ms:.2f}"
        )


def main():
    args = parse_args()
    k_values = parse_int_list(args.k_values)
    chunk_sizes = parse_int_list(args.chunk_sizes)
    overlaps = parse_int_list(args.overlaps)

    benchmark_queries = load_benchmark(args.benchmark)
    if not benchmark_queries:
        raise ValueError("Benchmark file has no queries.")

    docs = load_pdfs(args.pdf_dir)
    if not docs:
        raise ValueError(f"No PDFs found in {args.pdf_dir}.")

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

    reranker = None
    if args.profile == "rerank":
        reranker = CrossEncoderReranker(args.reranker_model)

    effective_candidate_k = max(args.candidate_k, max(k_values))
    sweep_results: list[ChunkingSweepResult] = []

    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= chunk_size:
                print(
                    f"Skipping invalid config chunk_size={chunk_size}, overlap={overlap}"
                )
                continue

            print(
                f"\n=== Config: chunk_size={chunk_size}, overlap={overlap}, "
                f"profile={args.profile} ==="
            )
            chunks = chunk_docs(
                docs,
                max_tokens=chunk_size,
                overlap_tokens=overlap,
            )
            chunk_stats = analytics(chunks)
            embedded_chunks = embed_chunks(chunks)
            index = build_faiss_index(embedded_chunks)
            retriever = FaissRetriever(index, chunks)

            page_rankings: list[list[str]] = []
            doc_rankings: list[list[str]] = []
            page_labels: list[set[str]] = []
            doc_labels: list[set[str]] = []
            latencies_ms: list[float] = []

            for benchmark_query in benchmark_queries:
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
                latencies_ms.append((time.perf_counter() - started_at) * 1000.0)

                page_rankings.append(build_ranked_ids(ranking.ranked_chunks, "page"))
                doc_rankings.append(build_ranked_ids(ranking.ranked_chunks, "doc"))
                page_labels.append(benchmark_query.relevant_pages)
                doc_labels.append(benchmark_query.relevant_doc_ids)

            page_metrics = compute_metric_bundle(page_rankings, page_labels, k_values)
            doc_metrics = compute_metric_bundle(doc_rankings, doc_labels, k_values)

            print(
                f"num_chunks={chunk_stats.num_chunks}"
                f" | avg_tokens={chunk_stats.avg_tokens:.1f}"
                f" | median_tokens={chunk_stats.median_tokens:.1f}"
                f" | p95_tokens={chunk_stats.p95_tokens:.1f}"
                f" | avg_latency_ms={statistics.mean(latencies_ms):.2f}"
            )
            print_metric_bundle("Page-level", page_metrics)
            print_metric_bundle("Doc-level", doc_metrics)

            sweep_results.append(
                ChunkingSweepResult(
                    profile=args.profile,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    num_chunks=chunk_stats.num_chunks,
                    avg_tokens=chunk_stats.avg_tokens,
                    median_tokens=chunk_stats.median_tokens,
                    p95_tokens=chunk_stats.p95_tokens,
                    avg_latency_ms=statistics.mean(latencies_ms),
                    page_metrics=metric_to_dict(page_metrics),
                    doc_metrics=metric_to_dict(doc_metrics),
                )
            )

    print_leaderboard(sweep_results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": args.benchmark,
            "profile": args.profile,
            "candidate_k": effective_candidate_k,
            "threshold": args.threshold,
            "results": [asdict(result) for result in sweep_results],
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote sweep summary to {output_path}")


if __name__ == "__main__":
    main()

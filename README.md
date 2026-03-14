# Local RAG Assistant

This repository is a local retrieval-augmented generation (RAG) assistant for PDF documents. It ingests PDFs, chunks and embeds them, stores the vectors in FAISS, reranks the best retrieval candidates with a cross-encoder, and answers questions with a local `llama.cpp` model.

It supports two query modes:

- `vanilla`: a straightforward retrieve-then-generate pipeline
- `agentic`: a LangGraph workflow with routing, retrieval, review, and answer generation nodes

There is also a small Streamlit UI for running queries in the browser.

## What the project does

1. Load PDFs from `data/raw/pdfs`
2. Split them into token-aware chunks
3. Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`
4. Store vectors and chunk metadata in FAISS
5. Rerank top retrieval candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`
6. Answer questions with a local GGUF model through `llama_cpp_python`

## Project layout

```text
.
├── data/
│   ├── raw/pdfs/              # source PDFs
│   └── faiss/                 # generated FAISS index + chunk metadata
├── src/
│   ├── agents/                # agent nodes used in the LangGraph flow
│   ├── app/                   # runtime and runner helpers
│   ├── generation/            # prompt builders
│   ├── ingestion/             # PDF loading + chunking
│   ├── llm/                   # local llama.cpp-backed LLM wrapper
│   ├── orchestration/         # LangGraph state + graph definition
│   ├── pipeline/              # vanilla retrieval/context pipelines
│   └── retrieval/             # embeddings, FAISS storage, retriever logic
├── ingest.py                  # build/rebuild the FAISS index
├── main.py                    # terminal chat interface
└── streamlit_app.py           # browser UI
```

## Prerequisites

- Python 3.9+
- A working local Python environment
- The local GGUF model expected by the app:
  `llama.cpp/build/models/qwen2.5-7b-instruct-q5_k_m.gguf`
- PDF files in `data/raw/pdfs/`

Notes:

- The embedding model (`all-MiniLM-L6-v2`) is downloaded by `sentence-transformers` on first use if it is not already cached.
- The reranker model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is also loaded on demand. If it is unavailable, the app falls back to FAISS retrieval order.
- The repository currently already contains sample PDFs under `data/raw/pdfs/` and a FAISS index under `data/faiss/`.
- If your GGUF model lives somewhere else, update `MODEL_PATH` in `src/llm/local_llm.py`.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Build or rebuild the index

If you add or replace PDFs, rebuild the FAISS index:

```bash
python3 ingest.py
```

The ingestion script now defaults to the best chunking setup from the current sweep:

- `--chunk-size 256`
- `--overlap 64`

You can still override either value explicitly if you want to test other configurations:

```bash
python3 ingest.py --chunk-size 384 --overlap 64
```

This script:

- loads every PDF page from `data/raw/pdfs/`
- chunks the text
- embeds the chunks
- writes the index to `data/faiss/index.faiss`
- writes chunk metadata to `data/faiss/index.chunks.pkl`

## Run the assistant in the terminal

Vanilla mode:

```bash
python3 main.py --mode vanilla
```

Agentic mode:

```bash
python3 main.py --mode agentic
```

Once the app is ready, type a question at the prompt. Type `quit()` to exit.

## Run the Streamlit UI

From the repository root:

```bash
python3 -m streamlit run streamlit_app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

The UI lets you:

- choose `vanilla` or `agentic`
- submit a question
- view the answer
- inspect the selected route and revision count

## How the two modes differ

### Vanilla mode

`vanilla` uses the retrieval pipeline directly:

1. Optionally decompose the query
2. Optionally expand the query
3. Retrieve candidate chunks from FAISS
4. Rerank the best candidates with a cross-encoder
5. Pack the highest-ranked context into a prompt
6. Generate one final answer

### Agentic mode

`agentic` compiles a LangGraph state machine in `src/orchestration/graph.py` with these nodes:

- `router`: decides whether the question can be answered directly or needs retrieval
- `direct_responder`: handles small talk and stable general-knowledge questions
- `extractor`: retrieves candidate context from the vector store
- `reviewer`: checks whether the retrieved context is relevant enough
- `summarizer`: generates the final answer from retrieved context

The router uses a mix of lightweight heuristics, indexed document-title aliases, and an LLM routing pass. That means queries mentioning titles already present in the local corpus can bias toward retrieval without forcing retrieval for unrelated public-source questions.

If retrieval is not good enough, the graph loops back through the extractor and reviewer up to the configured revision cap before forcing generation.

## Retrieval tuning

The runtime exposes a few optional environment variables:

- `RAG_ENABLE_RERANKER=0` disables cross-encoder reranking
- `RAG_CANDIDATE_K=24` controls how many deduplicated FAISS hits are kept before reranking
- `RAG_FINAL_K=6` controls how many chunks survive into the final prompt
- `RAG_RETRIEVAL_THRESHOLD=0.35` controls the FAISS similarity floor
- `RAG_AGENTIC_MAX_REVISIONS=2` caps extractor passes in the agentic workflow
- `RAG_AGENTIC_REVIEWER_ACCEPT_SCORE=3.0` skips the LLM reviewer for high-confidence reranked hits
- `RAG_AGENTIC_FORCE_RETRIEVAL=1` forces retrieval in agentic mode for document-grounded evaluations

The reranker improves context precision but adds query-time cost, so the first query can be slower while the model is loaded.

## Benchmarking Retrieval

If you want defensible retrieval statistics for your CV, create a small labeled benchmark and compare `faiss` against `rerank` on the same query set.

Recommended setup:

- Start with `20-50` manually written questions that reflect the kinds of queries you want the assistant to answer.
- Label the relevant `pages` first, because page-level labels are more stable than chunk IDs if you later change chunk size or overlap.
- Add `chunk_ids` as a stricter label set once your chunking strategy is settled.
- Add lightweight `answer_patterns` if you want to benchmark final answer quality, not just retrieval quality.
- Keep the benchmark fixed while you compare retrieval variants.

You can copy the template in `experiments/eval_queries/benchmark_template.json` and fill in your own labels, or edit `experiments/eval_queries/single_hop.json` directly.

Run the evaluation script from the repository root:

```bash
python3 experiments/retrieval_eval.py --benchmark experiments/eval_queries/single_hop.json
```

Useful options:

- `--show-top-k 5` prints the top retrieved chunks for each query to help with labeling and error analysis
- `--candidate-k 30` increases the number of FAISS hits considered before reranking
- `--use-expander` evaluates retrieval with query expansion enabled
- `--use-decomposer` evaluates retrieval with query decomposition enabled
- `--output-json experiments/results/retrieval_eval.json` saves the full summary for later comparison

The script reports:

- `precision@k`
- `recall@k`
- `hit-rate@k`
- `MRR`
- average query latency

It computes chunk-level, page-level, and doc-level metrics when the corresponding labels are present, then prints the delta between `faiss` and `rerank`.

## Mode Comparison

If you want to compare full assistant behavior, use the mode comparison script instead of the retrieval-only benchmark. This is the better fit for questions like "is `agentic` with reranking actually better than `vanilla` without reranking?"

Run it like this:

```bash
python3 experiments/mode_compare.py --benchmark experiments/eval_queries/single_hop.json --profiles vanilla_no_rerank,agentic_rerank
```

Useful options:

- `--show-failures 5` prints a few mismatched answers for quick error analysis
- `--limit 5` runs only the first few benchmark queries as a smoke test
- `--output-json experiments/results/mode_compare.json` saves the per-query results and profile summaries
- `--use-expander` and `--use-decomposer` enable those retrieval steps for every compared profile
- `--max-revisions 2` keeps agentic retries bounded to one adaptive retry after the first pass
- `--reviewer-accept-score 3.0` skips the expensive reviewer call on strong reranked results
- `--force-retrieval` evaluates retrieval-first behavior without changing the default interactive app

The script reports:

- answer match rate over queries with `answer_patterns`
- abstain rate and false-abstain rate
- average query latency
- average revision count
- route distribution (`direct` vs `retrieve`)

For a cleaner ablation, compare all four profiles:

```bash
python3 experiments/mode_compare.py --profiles vanilla_no_rerank,vanilla_rerank,agentic_no_rerank,agentic_rerank
```

That makes it easier to separate the effect of reranking from the effect of agent orchestration.

## Chunking Sweep

Chunk size and overlap are worth testing because they often move retrieval quality more than prompt tweaks do. For chunking comparisons, prefer the page-level metrics because page labels stay valid even when chunk boundaries change.

Run a chunking sweep like this:

```bash
python3 experiments/chunking_sweep.py --benchmark experiments/eval_queries/single_hop.json
```

Useful options:

- `--profile rerank` benchmarks your reranked pipeline for each chunking configuration
- `--chunk-sizes 256,384,512,768` tries several chunk lengths
- `--overlaps 0,32,64,96` tries several overlap settings
- `--output-json experiments/results/chunking_sweep.json` saves the sweep results

The sweep rebuilds chunked corpora and FAISS indices in memory, so it does not overwrite the main index under `data/faiss/`.

## Important files

- `main.py`: terminal interface
- `streamlit_app.py`: browser interface
- `ingest.py`: ingestion and indexing entrypoint
- `src/orchestration/graph.py`: LangGraph definition
- `src/app/runners.py`: `vanilla` and `agentic` runner implementations
- `src/app/runtime.py`: shared runtime builder
- `src/llm/local_llm.py`: local model wrapper and GGUF path

## Troubleshooting

### `ModuleNotFoundError`

Install the dependencies in your active environment:

```bash
python3 -m pip install -r requirements.txt
```

### Model file not found

Make sure this file exists:

```text
llama.cpp/build/models/qwen2.5-7b-instruct-q5_k_m.gguf
```

If your model is stored elsewhere, change `MODEL_PATH` in `src/llm/local_llm.py`.

### The app says "I don't know based on the provided context."

That usually means one of these is true:

- the relevant information is not present in the ingested PDFs
- the FAISS index is stale and needs to be rebuilt
- retrieval did not find chunks above the configured threshold

Try adding the right PDFs and rerunning:

```bash
python3 ingest.py
```

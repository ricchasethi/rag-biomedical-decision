# CLAUDE.md — BioRAG Decision-Support System

This file tells Claude how to work with this codebase effectively.

---

## Project Overview

BioRAG is a **retrieval-augmented generation engine** for scientific and biomedical documents,
designed as a decision-support system. The emphasis is on auditability and calibrated
uncertainty — every answer includes an explicit reasoning chain, evidence classification,
and knowledge gap analysis.

The core engine (`core/rag_engine.py`) is **pure Python stdlib with zero external
dependencies**. Keep it that way unless there is a compelling reason to add one.

---

## Running the Project

```bash
# Run tests (always do this before making changes)
python tests/test_biorag.py

# Interactive CLI
python cli.py

# Single query
python cli.py --query "What biomarkers predict CVD risk in T2DM?"

# Use Claude as answer synthesizer (requires: pip install anthropic)
python cli.py --llm --query "What biomarkers predict CVD risk in T2DM?"

# Use Claude and print the full prompt sent to the LLM before each answer
python cli.py --llm --show-prompt --query "What biomarkers predict CVD risk in T2DM?"

# Demo all preset queries
python cli.py --demo

# Hybrid retrieval: BM25 + dense (Qdrant) fused via RRF (requires: pip install qdrant-client sentence-transformers)
python cli.py --hybrid --query "What biomarkers predict CVD risk in T2DM?"
python cli.py --hybrid --llm --query "What plasma proteins predict Alzheimer's?"

# Ingest PubMed/PMC papers then enter interactive mode (requires: pip install requests)
python cli.py --ingest "alzheimer's disease biomarkers"
python cli.py --ingest "alzheimer's disease biomarkers" --ingest-max 25

# Ingest and persist new papers to data/sample_corpus.py (idempotent — skips existing IDs)
python cli.py --ingest "alzheimer's disease biomarkers" --ingest-max 25 --save-corpus

# Ingest into a hybrid index (chunks flow to both BM25 and Qdrant)
python cli.py --hybrid --ingest "alzheimer's disease biomarkers" --ingest-max 25

# Cross-encoder reranking: final semantic ranking stage (requires: pip install sentence-transformers)
python cli.py --rerank --query "What plasma proteins predict Alzheimer's?"
python cli.py --hybrid --rerank --query "What biomarkers predict CVD risk in T2DM?"

# Run the ingestion script standalone (supports --query, --max-results, --save-corpus, --hybrid)
python ingestion_pubmed.py --query "alzheimer's disease biomarkers" --max-results 10 --save-corpus

# Run retrieval evals (MRR + NDCG@K for BM25 vs reranker)
python evals/retrieval_eval.py
python evals/retrieval_eval.py --alzheimer-only --verbose

# Four-mode retrieval eval: BM25 / Dense / Hybrid (RRF) / Hybrid+Rerank
python evals/retrieval_eval.py --hybrid --alzheimer-only --verbose

# Add a fifth column comparing the cross-encoder final ranking (implies --hybrid)
python evals/retrieval_eval.py --cross-encoder --alzheimer-only

# Run hybrid retrieval tests
python tests/test_hybrid_retrieval.py

# Run cross-encoder reranker tests
python tests/test_cross_encoder_rerank.py

# Start API server (requires: pip install fastapi uvicorn pydantic)
python server.py
# → http://localhost:8000/docs

# Start API server with hybrid retrieval enabled
BIORAG_HYBRID=1 python server.py

# Start API server with cross-encoder reranking (combine with BIORAG_HYBRID for the full pipeline)
BIORAG_RERANK=1 python server.py
BIORAG_HYBRID=1 BIORAG_RERANK=1 python server.py

# Start MCP server standalone (requires: pip install "mcp[cli]")
python mcp_server.py

# Register MCP server with Claude Code (run once)
claude mcp add biorag python /absolute/path/to/mcp_server.py

# --- Scheduled arXiv ingestion pipeline (Airflow + Postgres + Qdrant) ---
# Bring up the stack (Airflow UI on :8080, Qdrant on :6333)
cd airflow && docker compose up -d

# Any stage standalone, inside the worker (repo is mounted at /opt/biorag)
docker compose exec airflow-worker bash -lc 'cd /opt/biorag && python -m biorag_pipeline.embed --stats'

# Pipeline test suites (need psycopg2 + a live biorag database)
docker compose exec airflow-worker python /opt/biorag/tests/test_arxiv_client.py
docker compose exec airflow-worker python /opt/biorag/tests/test_pipeline_db.py
```

---

## Architecture

> "Pipeline" means two things in this repo. This section and **Modifying the Pipeline** describe
> the **query** pipeline inside `core/rag_engine.py`. The **ingestion** pipeline — the scheduled
> arXiv DAG — is a separate system documented under
> [Scheduled Ingestion Pipeline](#scheduled-ingestion-pipeline-biorag_pipeline--airflow).

The query pipeline runs in this order. Each stage is a separate class in `core/rag_engine.py`:

```
BioRAGEngine.query(question)
  │
  ├─ QueryAnalyzer.analyze()          → intent, entities, expanded tokens
  ├─ InvertedIndex.search()           → BM25 retrieval, top-K chunks
  ├─ DenseRetriever.search()          → Qdrant cosine ANN top-K  ┐  (only when
  ├─ reciprocal_rank_fusion()         → fuse BM25 + dense via RRF ┘   dense_retriever set)
  ├─ Reranker.rerank()                → section-aware score adjustment + discriminative-token recall penalty
  │                                     (becomes a pre-filter to cross_encoder_candidates when cross_encoder set)
  ├─ CrossEncoderReranker.rerank()    → semantic (query,chunk) scoring, final top-K  (only when cross_encoder set)
  ├─ EvidenceClassifier.classify()    → direct / indirect / contradictory
  ├─ KnowledgeGapDetector.detect()    → missing data, contradictions, low relevance
  ├─ AnswerSynthesizer.synthesize()   → answer text + reasoning chain + confidence
  └─ FollowUpGenerator.generate()     → suggested follow-up questions
```

The dense-retrieval stage is **optional and injected** (like `ClaudeSynthesizer`). When
`dense_retriever` is `None` (the default), the pipeline is pure BM25 and `core/rag_engine.py`
imports nothing external. When set, BM25 and dense results are merged by RRF before reranking;
the fused score lands in `RetrievedChunk.score`, so every downstream stage is unaffected. See
the **Hybrid Retrieval** section for details.

### Key data structures

- `Chunk` — a chunked piece of a document with tokens, section, page
- `RetrievedChunk` — a `Chunk` with a retrieval score and rank (BM25, or the fused RRF score in hybrid mode)
- `EvidenceNode` — a classified chunk with relevance score and support type
- `ReasoningStep` — one step in the reasoning chain with a confidence score
- `DecisionOutput` — the full structured output returned by `BioRAGEngine.query()`

---

## Code Conventions

- **Type hints everywhere.** All functions must have full type annotations.
- **Dataclasses for data structures.** Do not use plain dicts for structured data passed between components.
- **No mutation of input arguments.** Each pipeline stage returns new objects.
- **Docstrings on all public methods.** One-line summary + explanation of non-obvious behavior.
- **No `print()` in library code.** Use the CLI layer (`cli.py`) for terminal output.
- **BM25 constants** (`K1`, `B`) are class-level attributes on `InvertedIndex` — adjust there, not inline.

---

## MCP Server

`mcp_server.py` wraps the BioRAG engine as an MCP server using FastMCP, so
Claude Code (and Claude Desktop) can call the engine as tools directly inside
a conversation.

### Tools exposed

| Tool | Signature | Description |
|---|---|---|
| `query` | `query(question: str) → dict` | Full RAG pipeline — returns answer, confidence, evidence nodes, knowledge gaps, follow-up questions |
| `ingest` | `ingest(pubmed_query: str, max_results: int = 10) → dict` | Fetch PubMed/PMC papers and add to the running corpus; returns updated corpus stats |
| `corpus_stats` | `corpus_stats() → dict` | Document/chunk/term counts with full-text vs abstract-only breakdown |

### Registration

```bash
# Install the SDK
pip install "mcp[cli]"

# Register with Claude Code (user-level, persists across projects)
claude mcp add biorag python /absolute/path/to/mcp_server.py

# Or project-scoped — add to .claude/settings.json:
# {
#   "mcpServers": {
#     "biorag": { "command": "python", "args": ["/abs/path/mcp_server.py"] }
#   }
# }
```

### Verify and use

```bash
claude mcp list      # biorag: python ... ✓ Connected
```

Then in a new Claude Code session:

```
Use biorag to query: what biomarkers predict Alzheimer's disease?
Use biorag to ingest 20 papers on CRISPR cancer therapy
Use biorag corpus_stats to show the current corpus
```

### Important notes

- MCP tools are injected at **session startup** — restart Claude Code after
  first registration to make the tools available.
- The engine loads `SAMPLE_DOCUMENTS` on startup. Documents added via `ingest`
  persist for the lifetime of the server process only (in-memory index).
- The server uses `stdio` transport — Claude Code starts it as a child process
  automatically; you do not need to run it manually.

---

## PubMed / PMC Ingestion

`ingestion_pubmed.py` fetches papers from NCBI and indexes them into a
`BioRAGEngine`.  The pipeline for each search result is:

1. **PubMed search** — `esearch` returns a list of PMIDs.
2. **PMC link resolution** — `elink` maps each PMID to a PMC ID (if the
   article has open-access full text).
3. **Full-text fetch** — `efetch db=pmc` retrieves JATS XML; the parser
   strips inline citation markers (`<xref ref-type="bibr">`) before text
   extraction, then extracts all `<sec>` / `<p>` elements with section headings.
4. **Abstract fallback** — PMIDs without a PMC record are fetched from
   `efetch db=pubmed` and only the abstract is indexed.

Metadata stored per document: `source` (`"pubmed_central"` or `"pubmed"`),
`has_full_text` (bool), `pmcid`, `pmid`, `year`, `journal`.

`ingest_pubmed()` accepts an optional `engine` argument and a `save_corpus`
flag to persist results to `data/sample_corpus.py`:

```python
from ingestion_pubmed import ingest_pubmed
from core.rag_engine import BioRAGEngine

engine = BioRAGEngine()
ingest_pubmed("CRISPR cancer therapy", max_results=20, engine=engine)

# Persist new papers to sample_corpus.py (skips already-present doc IDs)
ingest_pubmed("alzheimer's disease biomarkers", max_results=25,
              engine=engine, save_corpus=True)
```

### Citation cleaning

Two layers strip citation number artifacts from ingested text:

1. **XML layer** (`_strip_citation_xrefs`): removes `<xref ref-type="bibr">` elements
   from JATS XML in-place before `itertext()` runs. Handles PMC full-text papers.

2. **Regex layer** (`TextProcessor.clean_text`): five patterns covering both no-space
   (`.7,8`, `,3`) and spaced (`. 1 Word`, `) 9 and`, ` 31 , 32 , 33 Word`, `, 37 and`)
   citation formats. Applied at chunking time, so already-stored corpus text is cleaned
   automatically on the next engine load.

Single inline citations without punctuation context (`eQTLGen 38 yielded`,
`50 million individuals`) are intentionally left untouched — they cannot be
distinguished from real measurements without NLP.

---

## Answer Quality Evals (LLM-as-Judge)

`evals/answer_eval.py` scores the prose answer produced by `BioRAGEngine.query()` against
hand-authored reference claims using Claude as an independent judge.

### Files

| File | Purpose |
|---|---|
| `evals/answer_ground_truth.py` | `ANSWER_CLAIMS`: 10 `AnswerClaim` objects covering all four corpus documents. Each claim has `reference_claim`, `expected_entities`, `expected_direction`, and `expected_context`. |
| `evals/answer_eval.py` | `AnswerEvaluator` class + CLI runner. Runs `engine.query()` per query, then calls Claude via `tool_use` to score the answer on eight rubric dimensions. |

### Rubric (per query, max 10 points)

| Dimension | Max | Description |
|---|---|---|
| Semantic Coverage | 2 | Does the answer address the right phenomenon (not just the topic area)? |
| Entity Coverage | 2 | Are the specific genes/markers/drugs named in the correct context? |
| Directional Agreement | 1 | Does the stated direction of effect match the claim (elevated/decreased/no effect)? |
| Quantitative Detail | 1 | Are magnitudes or statistics consistent with the claim? |
| Contextual Accuracy | 1 | Is the finding placed in the claim-specific context (timepoint, tissue, subgroup)? |
| Source Attribution | 1 | Are all factual claims linked to inline numbered citations ([1], [2], etc.)? |
| Evidence Strength | 1 | Is the study design / evidence type explicitly named (RCT, meta-analysis, cohort, in vitro)? |
| Uncertainty Calibration | 1 | Does the expressed confidence match the evidence quality (assertive for strong evidence, hedged for weak)? |

### Running

```bash
# All 10 reference claims
python evals/answer_eval.py

# Alzheimer's subset (4 claims)
python evals/answer_eval.py --alzheimer-only

# Full answer text + judge rationale per query
python evals/answer_eval.py --verbose

# Use ClaudeSynthesizer for answers (requires ANTHROPIC_API_KEY)
python evals/answer_eval.py --llm

# Side-by-side table: retrieval metrics (MRR/NDCG) vs answer quality scores
python evals/answer_eval.py --with-retrieval
```

### Key insight this eval reveals

The retrieval eval (MRR/NDCG) measures whether the right document was found.
The answer eval measures whether the answer said the right thing. A system can
score MRR=1.0 while still missing the direction of effect or omitting key entities
— the combined `--with-retrieval` table makes this visible.

### Adding new reference claims

Append to the relevant list in `evals/answer_ground_truth.py`:

```python
AnswerClaim(
    query_id="Q17",
    reference_claim="APOE4 carriers have a 3–4× increased risk of late-onset Alzheimer's disease.",
    expected_entities=["APOE4", "late-onset", "risk"],
    expected_direction="increased risk in APOE4 carriers",
    expected_context="late-onset Alzheimer's disease, genetic risk factor",
),
```

The `query_id` must match an entry in `evals/ground_truth.py`. Add the corresponding
`RetrievalQuery` there first if the query is new.

---

## Retrieval Evals

The `evals/` directory contains a harness that measures retrieval quality at two
pipeline stages independently — BM25 alone vs. BM25 + reranker — so improvements or
regressions are attributable to a specific stage.

### Files

| File | Purpose |
|---|---|
| `evals/ground_truth.py` | `EVAL_QUERIES`: 16 `RetrievalQuery` objects with hand-labelled `{doc_id: grade}` relevance dicts. 7 queries focus on Alzheimer's disease (`ALZHEIMER_QUERIES`). |
| `evals/retrieval_eval.py` | `RetrievalEvaluator` class + CLI runner. Aggregates chunk scores to document level (max-pooling), then computes MRR and NDCG@K. |

### Metrics

- **MRR@K** — reciprocal rank of the first relevant doc in top-K. Good for decision-support
  where the user stops at the first useful answer.
- **NDCG@K** — graded DCG (grade 2 = direct, grade 1 = partial, grade 0 = irrelevant).
  Penalises burying a high-relevance document deep in the list.

### Running

```bash
python evals/retrieval_eval.py                      # all 16 queries
python evals/retrieval_eval.py --alzheimer-only     # AD subset only
python evals/retrieval_eval.py --verbose            # per-query top-3 vs ground truth
python evals/retrieval_eval.py --ks 1 5 10          # custom K values
python evals/retrieval_eval.py --hybrid             # four-mode: BM25 / Dense / Hybrid / Hybrid+Rerank
```

The `--hybrid` flag swaps `print_report` for `print_hybrid_report`, comparing four modes
side by side so each stage's contribution is attributable (same philosophy as the BM25-vs-reranker
Δ column). It builds an in-memory Qdrant index via `build_hybrid_engine()`, so the eval always
reflects the current corpus. Requires `qdrant-client` + `sentence-transformers`.

### Maintaining the ground truth

After ingesting new papers, add entries to the relevant list in `evals/ground_truth.py`:

```python
RetrievalQuery(
    query_id="Q17",
    query="What is the role of APOE4 in Alzheimer's disease risk?",
    intent="mechanism",
    relevant_docs={
        "pmid_<new_doc_id>": 2,   # directly addresses the query
        "neuro_2026_001": 1,       # tangentially relevant
    },
),
```

The `EVAL_QUERIES` list at the bottom of the file imports all sub-lists — add your new
query to the appropriate sub-list and it will be picked up automatically.

### What the Δ column reveals

A negative NDCG@3/5 delta (Reranked − BM25) means the reranker's `rerank_top_k` budget
is cutting documents that are partially relevant to multi-document queries. The fix is
to raise `rerank_top_k` or adjust `Reranker.SECTION_WEIGHTS` for the affected intent.

---

## Hybrid Retrieval

`hybrid_retrieval.py` adds dense (embedding-based) retrieval alongside BM25 and fuses the
two with Reciprocal Rank Fusion (RRF). It is **optional and injected** via
`BioRAGEngine(dense_retriever=...)`, mirroring the `ClaudeSynthesizer` pattern —
`core/rag_engine.py` stays stdlib-only and imports `hybrid_retrieval` lazily inside
`query()`, only when a `dense_retriever` is present.

```python
from hybrid_retrieval import EmbeddingModel, DenseRetriever
from core.rag_engine import BioRAGEngine

engine = BioRAGEngine(dense_retriever=DenseRetriever(EmbeddingModel()))
result = engine.query("What plasma biomarkers predict Alzheimer's?")
```

### Components

| Component | Role |
|---|---|
| `EmbeddingModel` | Wraps sentence-transformers. Lazy-loads on first `encode()`; caches vectors by an MD5 of the text. Default model: `pritamdeka/S-PubMedBert-MS-MARCO` (768-dim, PubMed-tuned). |
| `DenseRetriever` | Owns a Qdrant collection. `add_chunks()` embeds + upserts; `search()` returns `[(chunk_id, cosine_score)]`. |
| `reciprocal_rank_fusion()` | Merges BM25 + dense rankings: `score = Σ 1/(k + rank_i)`, `k=60`. Returns `RetrievedChunk` list with the fused score. |

### How fusion stays transparent to the rest of the pipeline

After RRF, `RetrievedChunk.score` holds the fused score and the list feeds straight into
`Reranker.rerank()`. The existing 0–1 normalisation in `query()` (`r.score / max(max_score, …)`)
handles it, so `EvidenceNode.relevance_score`, the classifier, gaps, and synthesizer are all
unchanged. `match_terms` is preserved from the BM25 result; dense-only hits get an empty list.

### Qdrant persistence modes

| Mode | Config | Use case |
|---|---|---|
| File-based (default) | `DenseRetriever(model)` → `./qdrant_data` | Dev / single process; vectors survive restarts |
| In-memory | `DenseRetriever(model, qdrant_path=":memory:")` | Tests and evals (fresh each run) |
| Server | point `DenseRetriever.client` at `QdrantClient(url=...)` | Multi-process production |

`./qdrant_data/` is gitignored. On startup the engine re-adds `SAMPLE_DOCUMENTS`, but
`add_chunks()` asks Qdrant which point IDs already exist (deterministic `uuid5` per chunk id)
and embeds only genuinely new chunks — so the embedding model is hit once per chunk over the
collection's lifetime, not on every launch.

### Wiring across entry points

| Entry point | How to enable |
|---|---|
| `cli.py` | `--hybrid` flag |
| `ingestion_pubmed.py` | `--hybrid` flag (standalone) or `ingest_pubmed(..., dense_retriever=...)` |
| `server.py` | `BIORAG_HYBRID=1` env var |
| `evals/retrieval_eval.py` | `--hybrid` flag → four-mode comparison (BM25 / Dense / Hybrid / Hybrid+Rerank) |

### Tuning

- **RRF `rrf_k`** (default 60): lower it to amplify the top-rank bonus; raise it to flatten
  rankings. A chunk ranked 1st in both lists scores ≈ `2/(rrf_k+1)`.
- **Embedding model**: pass a different name to `EmbeddingModel(model_name=...)`. Use
  `fastembed`-compatible models (e.g. `BAAI/bge-small-en-v1.5`, 384-dim) to avoid pulling `torch`.
- **Candidate depth**: each retriever returns its own `retrieval_top_k` before fusion, so RRF
  sees up to `2 × retrieval_top_k` candidates.

> Note: `DenseRetriever.search()` uses qdrant-client's `query_points(query=...)`; the older
> `.search(query_vector=...)` API is deprecated in qdrant-client ≥ 1.18.

---

## Cross-Encoder Reranking

`cross_encoder_rerank.py` adds a **true semantic reranker** as the final ranking stage.
Where the lexical `Reranker` (in `core/rag_engine.py`) only scores term overlap, section
weights, and discriminative-token recall, a cross-encoder feeds the query and chunk text
through a transformer *together* and emits one calibrated relevance score per pair —
modelling paraphrase, negation, and entity disambiguation that BM25 cannot see.

Like `DenseRetriever` and `ClaudeSynthesizer`, it is **optional and injected** via
`BioRAGEngine(cross_encoder=...)`; `core/rag_engine.py` stays stdlib-only and never imports
the module except in spirit (the type is a forward-reference string annotation).

```python
from cross_encoder_rerank import CrossEncoderReranker
from core.rag_engine import BioRAGEngine

engine = BioRAGEngine(cross_encoder=CrossEncoderReranker())
result = engine.query("What plasma biomarkers predict Alzheimer's?")

# Combine with hybrid retrieval for the full pipeline
from hybrid_retrieval import EmbeddingModel, DenseRetriever
engine = BioRAGEngine(
    dense_retriever=DenseRetriever(EmbeddingModel()),
    cross_encoder=CrossEncoderReranker(),
)
```

### Two-stage reranking: lexical pre-filter → cross-encoder

When a `cross_encoder` is set, the pipeline becomes:

```
BM25 → dense → RRF → Reranker (pre-filter, top cross_encoder_candidates) → CrossEncoderReranker (final top rerank_top_k)
```

The lexical `Reranker` is **not discarded** — it stays upstream as a cheap pre-filter that
narrows the candidate set to `cross_encoder_candidates` (default 12) before the more
expensive cross-encoder runs. This bounds cross-encoder cost while letting the
discriminative-token penalty drop obvious off-topic chunks first. The cross-encoder then
produces the final `rerank_top_k` ranking.

### How it stays transparent to the rest of the pipeline

`CrossEncoderReranker.rerank()` writes a **sigmoid-mapped score in (0, 1)** to
`RetrievedChunk.score`. The existing 0–1 normalisation in `query()` handles it, so
`EvidenceNode.relevance_score`, the classifier, gaps, and synthesizer are all unchanged.
`match_terms` is carried over from the candidate. The knowledge-gap detector still uses the
raw BM25/RRF `all_results` for its IDF-based low-score threshold, so it is unaffected.

### Components

| Component | Role |
|---|---|
| `CrossEncoderReranker` | Wraps sentence-transformers `CrossEncoder`. Lazy-loads on first `rerank()`; caches pair scores by an MD5 of `query + chunk_text`. Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`. |

### Wiring across entry points

| Entry point | How to enable |
|---|---|
| `cli.py` | `--rerank` flag (combine with `--hybrid`) |
| `server.py` | `BIORAG_RERANK=1` env var (combine with `BIORAG_HYBRID=1`) |
| `evals/retrieval_eval.py` | `--cross-encoder` flag → adds a fifth `Hybrid+CE` column (implies `--hybrid`) |

### Tuning

- **`cross_encoder_candidates`** (default 12): the pre-filter budget. Raise it to give the
  cross-encoder more candidates (better recall, higher latency); lower it to cut cost.
- **Model**: pass a different name to `CrossEncoderReranker(model_name=...)`. For tighter
  biomedical fit, try a PubMedBERT cross-encoder such as `ncbi/MedCPT-Cross-Encoder` at the
  cost of a larger download.

### What the eval reveals

On the current 4-document sample corpus, the four-mode `--cross-encoder` eval shows
`Hybrid+CE` improving MRR@5 over BM25 (+0.02–0.05) but **trailing the hand-tuned lexical
`Hybrid+Rerank`**. This is expected and honest: with only 4 documents and document-level
max-pooling there is little to reorder, and `Reranker.SECTION_WEIGHTS` is tuned on exactly
these queries. A cross-encoder's advantage grows with corpus size/diversity and at the chunk
level, and a biomedical cross-encoder would likely close the gap. Re-run the eval after
ingesting a larger corpus before drawing conclusions.

---

## Adding a New Document to the Corpus

```python
from core.rag_engine import BioRAGEngine

engine = BioRAGEngine()
n_chunks = engine.add_document(
    doc_id="unique_id",           # must be unique across corpus
    title="Paper Title",
    text="Full document text...", # plain text, section headers detected automatically
    metadata={"year": 2024, "journal": "Nature"}
)
print(f"Indexed {n_chunks} chunks")
```

To add documents permanently to the sample corpus, append to `data/sample_corpus.py`
following the existing `SAMPLE_DOCUMENTS` list format.

---

## Modifying the Pipeline

### Changing chunk size or overlap
Pass `chunk_size` and `chunk_overlap` to `BioRAGEngine(...)`. Smaller chunks improve
precision; larger chunks preserve more context per retrieval hit. The overlap prevents
splitting key sentences across boundaries.

### Changing how many results are retrieved / reranked
`retrieval_top_k` controls BM25 candidate count; `rerank_top_k` controls how many
survive reranking. Increasing `retrieval_top_k` improves recall at the cost of more
reranking work.

### Reranker discriminative-token penalty

`Reranker._GENERIC_BIO_TERMS` is a frozenset of ~80 words that appear in virtually
every biomedical paper (biomarker, disease, patient, predict, detect, assess, elevated…).
These carry no signal about *which* disease or entity a paper covers.

At rerank time, **discriminative tokens** = `query_tokens − _GENERIC_BIO_TERMS`. A chunk
that matches none of them gets `entity_factor = 0.15` (85% penalty). A chunk that
matches some gets `entity_factor = 0.6 + 0.4 × (matched / total)`.

**Why this matters**: BM25 only rewards term presence, not absence. A lung cancer paper
that uses "biomarker" heavily can score near the top for an Alzheimer's query. The
penalty suppresses these false positives without touching BM25 or the index.

**Tuning**: if a legitimate paper is being under-scored, check whether its key tokens
accidentally appear in `_GENERIC_BIO_TERMS`. Conversely, if off-topic papers still
surface, add their shared generic tokens to the set.

### Adding a new query intent type
1. Add the intent name and trigger phrases to `QueryAnalyzer.QUERY_TYPES`.
2. Add section weights for that intent to `Reranker.SECTION_WEIGHTS`.
3. Add follow-up templates to `FollowUpGenerator.TEMPLATES`.

### Upgrading to embedding-based retrieval
Replace or augment `InvertedIndex.search()`. The reranker, classifier, synthesizer,
and gap detector are all retrieval-agnostic — they only care about the `RetrievedChunk`
interface. The shipped implementation of this is **hybrid retrieval** (`hybrid_retrieval.py`),
injected via `BioRAGEngine(dense_retriever=...)` — see the **Hybrid Retrieval** section.

### Using LLM answer generation
`llm_synthesizer.py` provides `ClaudeSynthesizer`, a ready-to-use subclass of
`AnswerSynthesizer` that calls `claude-sonnet-4-6` at `temperature=0`.

```python
from llm_synthesizer import ClaudeSynthesizer
from core.rag_engine import BioRAGEngine

engine = BioRAGEngine(synthesizer=ClaudeSynthesizer())
result = engine.query("What plasma biomarkers predict Alzheimer's?")

# Inspect the exact prompt sent to Claude
print(engine.synthesizer.last_prompt["system"])
print(engine.synthesizer.last_prompt["user"])
```

Key properties:
- `SYSTEM_PROMPT` is a module-level constant in `llm_synthesizer.py` — easy to audit and version.
- The user message formats each `EvidenceNode` as a numbered excerpt with title, section, relevance, and support type (DIRECT / INDIRECT / CONTRADICTORY).
- The system prompt forbids Claude from drawing on training knowledge; if the excerpts are insufficient it must say so.
- Prompt caching (`"cache_control": {"type": "ephemeral"}`) is enabled on the system prompt.
- Falls back to the parent's rule-based `_build_answer()` if the API call fails.

To write a custom synthesizer, subclass `AnswerSynthesizer`, override `synthesize()`,
and pass an instance to `BioRAGEngine(synthesizer=...)`. The reasoning chain steps and
confidence scoring are inherited from the parent class.

---

## Testing

All tests live in `tests/test_biorag.py` and use no test framework beyond `assert`.
Run with `python tests/test_biorag.py` — no pytest required (though pytest works fine too).

When adding a new component or modifying existing behavior, add tests in the
corresponding group in the test file. The test groups mirror the pipeline stages:
`TextProcessor`, `DocumentChunker`, `InvertedIndex`, `QueryAnalyzer`,
`EvidenceClassifier`, `End-to-End Pipeline`.

The end-to-end tests use the real sample corpus from `data/sample_corpus.py`.
Do not mock the corpus in end-to-end tests — the test queries are chosen to have
known relevant documents.

Hybrid retrieval has its own suite, `tests/test_hybrid_retrieval.py` (run with
`python tests/test_hybrid_retrieval.py`), covering `EmbeddingModel`, `DenseRetriever`,
`reciprocal_rank_fusion`, and engine integration. It uses a `FakeEmbeddingModel` and an
in-memory Qdrant collection so most tests stay fast and offline; only the two `EmbeddingModel`
tests load the real sentence-transformers model (once, to check dimension and caching).

The cross-encoder stage has `tests/test_cross_encoder_rerank.py` (13 checks). The ingestion
pipeline has two more suites that need `psycopg2` and, for one of them, a live database — see
**Pipeline tests** below. Five suites, 115 checks in total.

---

## Scheduled Ingestion Pipeline (`biorag_pipeline/` + `airflow/`)

A second, independent ingestion path from `ingestion_pubmed.py`. That one is in-process,
in-memory and PubMed-facing; this one is scheduled, durable and arXiv-facing, backed by
Postgres for state and a Qdrant **server** for vectors.

```
migrate → discover → fetch → parse (dynamically mapped) → chunk → embed → finish
```

### The seam that must not be crossed

This is the single most important convention in the pipeline:

- `biorag_pipeline/repository.py` knows nothing about `core.rag_engine`.
- `core/rag_engine.py` knows nothing about Postgres.
- The conversion between the engine's `Chunk` and the repository's `ChunkRecord` happens in
  `biorag_pipeline/chunk.py` **and nowhere else**.

Either side can be replaced without touching the other. Do not import `repository` into the
engine, and do not import engine internals into `repository`.

### Running the pipeline

The repo is mounted at `/opt/biorag` in every container, so edits take effect with no rebuild.

```bash
cd airflow
docker compose up -d
docker compose ps                       # all services healthy
docker compose logs -f airflow-worker

# Every stage is a standalone CLI — develop with Airflow stopped
docker compose exec airflow-worker bash -lc 'cd /opt/biorag && python -m biorag_pipeline.migrate --status'

python -m biorag_pipeline.discover --days 3 --max-results 10 --dry-run
python -m biorag_pipeline.discover --search "protein folding" --field ti --any-category
python -m biorag_pipeline.fetch    --limit 5
python -m biorag_pipeline.parse    --limit 10 --show      # section breakdown + text preview
python -m biorag_pipeline.chunk    --limit 10 --show      # per-section chunk counts
python -m biorag_pipeline.embed    --limit 500
python -m biorag_pipeline.embed    --stats                # Postgres vs Qdrant counts
python -m biorag_pipeline.embed    --search "QUERY"       # sanity-check the vector index

# The DAG registers paused by default
docker compose exec airflow-worker airflow dags list
docker compose exec airflow-worker airflow dags unpause arxiv_ingest_daily
docker compose exec airflow-worker airflow dags trigger arxiv_ingest_daily
```

Every stage accepts `--arxiv-id` (repeatable) to process specific papers rather than the
pending queue, and `--run-id` to attribute errors to a run.

### Stage responsibilities

| Stage | Module | Reads | Writes | Status transition |
|---|---|---|---|---|
| `migrate` | `migrate.py` | `migrations/*.sql` | `schema_migrations` | — |
| `discover` | `discover.py` | arXiv Atom API | `papers`, `ingest_runs` | → `discovered` |
| `fetch` | `fetch.py` | `papers` | `var/raw/*.pdf` | `discovered` → `fetched` |
| `parse` | `parse.py` | PDFs | `var/raw/*.txt` | `fetched` → `parsed` |
| `chunk` | `chunk.py` | `.txt` files | `chunks` | `parsed` → `chunked` |
| `embed` | `embed.py` | `chunks` | Qdrant | `chunked` → `indexed` |
| `finish` | DAG-local | `ingest_runs` | `ingest_runs` | — |

### Invariants to preserve when editing

- **Status is the work queue.** No stage takes its work list from the previous stage's XCom —
  each queries Postgres by status. Keep it that way: it is what makes a half-failed run
  resumable. Never pass a list of ids between tasks.
- **Postgres says what *should* exist; `chunks.embedded` says what *does*.** Papers reach
  `indexed` via `repo.promote_indexed()` — a database predicate — not because `embed_pending()`
  believes it succeeded. Do not shortcut this.
- **One bad paper must never fail the batch.** Failures are recorded as data in
  `ingest_errors` and the paper is marked `failed`; exceptions are for programmer errors only.
  This mirrors the engine rule about not raising on low-quality results.
- **Network calls never happen inside a database transaction.** `discover` opens a connection
  to record the run, closes it, searches arXiv, then opens a second connection for the writes.
  A minute-long transaction blocks vacuum and risks idle-in-transaction timeouts.
- **IDs are deterministic.** Chunk ids are `md5(doc_id:char_offset)[:12]`; Qdrant point ids are
  `uuid5(NAMESPACE_DNS, chunk_id)`. Re-processing identical text overwrites in place. Do not
  introduce random or sequential ids.
- **`sanitize_text()` runs on both write and read.** A single NUL byte from a PDF makes
  Postgres reject an entire batch insert. `chunk.py` re-sanitises on read because `.txt` files
  written before the parser learned to strip control characters still contain them.
- **Heavy imports live inside task bodies**, never at DAG module level. The dag-processor
  re-imports the DAG file every few seconds; importing torch there would make every parse cycle
  take seconds.
- **`fetch` is deliberately not dynamically mapped.** The arXiv throttle is process-local, so
  parallel mapped instances would each start their own timer and hammer the API. `parse` *is*
  mapped, in batches of `PARSE_BATCH_SIZE = 5`, because pypdf extraction is CPU-bound with no
  external rate limit.

### Database schema

Four tables plus `schema_migrations`. `papers.arxiv_id` is the **natural** primary key, which
is what makes ingestion idempotent — "have I seen this paper?" is a uniqueness constraint
enforced by Postgres, not application logic that can drift.

```
discovered → fetched → parsed → chunked → indexed
                  ↘ failed     ↘ skipped
```

`chunks` mirrors `core.rag_engine.Chunk` **minus `tokens`** — those are a pure function of
`TextProcessor.tokenize()` and would go stale the next time `clean_text()` changes. Store the
count only. `chunks_pending_idx` is a partial index on unembedded rows, so the embedding queue
stays small regardless of corpus size.

### Adding a migration

Add a numbered `.sql` file to `biorag_pipeline/migrations/`. They are applied in filename
order, each in its own transaction, recorded in `schema_migrations` — so re-running is a no-op
and `migrate` is safe as the first task of every DAG run. Write them re-runnably
(`DROP CONSTRAINT IF EXISTS` before `ADD CONSTRAINT`), since an earlier attempt may have got
part-way. Do not reach for Alembic: Airflow already runs its own Alembic instance against the
`airflow` database, and a second framework buys nothing for a four-table schema.

### Adding a pipeline stage

1. Write `biorag_pipeline/<stage>.py` exposing `<stage>_pending(run_id, limit, ...) -> <Stage>Result`,
   with an `argparse` CLI under `if __name__ == "__main__"`.
2. Add any new status value to the `papers_status_check` constraint via a migration.
3. Add a thin `@task` wrapper to the DAG that calls it and returns a small dict of counters.
4. Query Postgres by status inside the stage — never accept a work list as an argument
   (`--arxiv-id` for manual runs is the exception).

### Configuration

All settings come from the environment via `biorag_pipeline/config.py`, with host-friendly
defaults so the same code runs inside Airflow or from a shell. Container values are set in
`airflow/docker-compose.override.yml`.

| Variable | Default | Purpose |
|---|---|---|
| `BIORAG_DB_URL` | `postgresql://airflow:airflow@localhost:5432/biorag` | Pipeline database |
| `BIORAG_QDRANT_URL` | `http://localhost:6333` | Qdrant server |
| `BIORAG_QDRANT_COLLECTION` | `biorag_chunks` | Collection name |
| `BIORAG_EMBED_MODEL` | `pritamdeka/S-PubMedBert-MS-MARCO` | Embedding model |
| `BIORAG_RAW_DIR` | `./var/raw` | PDFs and extracted text |
| `BIORAG_ARXIV_CATEGORIES` | `q-bio.QM,q-bio.GN,q-bio.NC` | Default categories |
| `BIORAG_ARXIV_MAX_RESULTS` | `50` | Per-run cap |
| `BIORAG_CHUNK_SIZE` / `BIORAG_CHUNK_OVERLAP` | `512` / `64` | Mirrors `BioRAGEngine` defaults |

`BIORAG_DB_URL` is written in SQLAlchemy form (`postgresql+psycopg2://`) so the same variable
can drive SQLAlchemy later; `db.dsn()` strips the `+driver` suffix, which libpq rejects.

Changing `BIORAG_EMBED_MODEL` to a different dimension requires a **new collection** — Qdrant
vector dimensions are immutable. `QdrantIndexer.ensure_collection()` probes the dimension from
the model rather than hardcoding it, so only the collection name needs to change.

### Pipeline tests

```bash
docker compose exec airflow-worker python /opt/biorag/tests/test_arxiv_client.py   # 32 checks, offline
docker compose exec airflow-worker python /opt/biorag/tests/test_pipeline_db.py    # 28 checks, live DB
```

`test_arxiv_client.py` is offline — it parses fixture XML. `test_pipeline_db.py` runs against
the live `biorag` database under a reserved id (`0000.99999`) and cleans up after itself.

Its `pending_queue()` helper exists for a reason: `repo.pending_chunks()` is a **corpus-wide**
queue by design, so asserting on its raw length only holds against an empty database. Scope
new assertions to `TEST_ID` the same way, or the suite will pass locally and fail on any
database holding real papers.

### What not to do here

- Do not import `hybrid_retrieval` or `cross_encoder_rerank` into `biorag_pipeline/repository.py`
  or `db.py` — the storage layer stays free of ML dependencies.
- Do not add an ORM. The schema is four tables and the queries are simple; an ORM would only
  add a dependency that has to stay compatible with whatever Airflow pins.
- Do not put pipeline logic in the DAG file. Every task is a wrapper that calls one function in
  `biorag_pipeline/` and returns a small dict — that is what keeps the pipeline testable from a
  shell with Airflow stopped, and keeps DAG parsing fast.
- Do not remove the `+cpu` torch pin in `airflow/Dockerfile`. Without it, sentence-transformers
  drags in the CUDA build and the image goes from ~4.8 GB to ~9 GB.
- Do not commit `airflow/.env`, `airflow/logs/`, or `var/` — all gitignored runtime state.

---

## What Not to Do

- Do not add external dependencies to `core/rag_engine.py`. It must stay stdlib-only.
- Do not raise exceptions inside pipeline stages for low-quality results — return
  a `DecisionOutput` with low confidence instead. Exceptions are for programmer errors only.
- Do not hardcode document IDs or titles anywhere in the engine logic.
- Do not add LLM calls to the core engine (`core/rag_engine.py`). LLM integration lives
  in `llm_synthesizer.py` and is injected via `BioRAGEngine(synthesizer=ClaudeSynthesizer())`.
- Do not return raw BM25 scores to the user — they are not interpretable. Always
  normalize to a 0–1 relevance score before surfacing in `EvidenceNode`.
- Do not `import hybrid_retrieval` at the top of `core/rag_engine.py`. It pulls in
  `qdrant-client` + `sentence-transformers`; keep the import lazy inside `query()` and the
  `dense_retriever` parameter a forward-reference string annotation so the core stays stdlib-only.
- Do not import `biorag_pipeline` (or psycopg2) into `core/rag_engine.py`. The engine must not
  know that Postgres exists; the pipeline depends on the engine, never the reverse.
- Likewise do not `import cross_encoder_rerank` in `core/rag_engine.py`. It pulls in
  `sentence-transformers`; the `cross_encoder` parameter is a forward-reference string
  annotation and the injected object's `.rerank()` is called directly — no import needed.

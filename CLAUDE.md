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

# Run the ingestion script standalone (supports --query, --max-results, --save-corpus, --hybrid)
python ingestion_pubmed.py --query "alzheimer's disease biomarkers" --max-results 10 --save-corpus

# Run retrieval evals (MRR + NDCG@K for BM25 vs reranker)
python evals/retrieval_eval.py
python evals/retrieval_eval.py --alzheimer-only --verbose

# Four-mode retrieval eval: BM25 / Dense / Hybrid (RRF) / Hybrid+Rerank
python evals/retrieval_eval.py --hybrid --alzheimer-only --verbose

# Run hybrid retrieval tests
python tests/test_hybrid_retrieval.py

# Start API server (requires: pip install fastapi uvicorn pydantic)
python server.py
# → http://localhost:8000/docs

# Start API server with hybrid retrieval enabled
BIORAG_HYBRID=1 python server.py

# Start MCP server standalone (requires: pip install "mcp[cli]")
python mcp_server.py

# Register MCP server with Claude Code (run once)
claude mcp add biorag python /absolute/path/to/mcp_server.py
```

---

## Architecture

The pipeline runs in this order. Each stage is a separate class in `core/rag_engine.py`:

```
BioRAGEngine.query(question)
  │
  ├─ QueryAnalyzer.analyze()          → intent, entities, expanded tokens
  ├─ InvertedIndex.search()           → BM25 retrieval, top-K chunks
  ├─ DenseRetriever.search()          → Qdrant cosine ANN top-K  ┐  (only when
  ├─ reciprocal_rank_fusion()         → fuse BM25 + dense via RRF ┘   dense_retriever set)
  ├─ Reranker.rerank()                → section-aware score adjustment + discriminative-token recall penalty
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

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

# Demo all preset queries
python cli.py --demo

# Ingest PubMed/PMC papers then enter interactive mode (requires: pip install requests)
python cli.py --ingest "alzheimer's disease biomarkers"
python cli.py --ingest "alzheimer's disease biomarkers" --ingest-max 25

# Ingest and persist new papers to data/sample_corpus.py (idempotent — skips existing IDs)
python cli.py --ingest "alzheimer's disease biomarkers" --ingest-max 25 --save-corpus

# Run the ingestion script standalone (supports --query, --max-results, --save-corpus)
python ingestion_pubmed.py --query "alzheimer's disease biomarkers" --max-results 10 --save-corpus

# Run retrieval evals (MRR + NDCG@K for BM25 vs reranker)
python evals/retrieval_eval.py
python evals/retrieval_eval.py --alzheimer-only --verbose

# Start API server (requires: pip install fastapi uvicorn pydantic)
python server.py
# → http://localhost:8000/docs

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
  ├─ Reranker.rerank()                → section-aware score adjustment
  ├─ EvidenceClassifier.classify()    → direct / indirect / contradictory
  ├─ KnowledgeGapDetector.detect()    → missing data, contradictions, low relevance
  ├─ AnswerSynthesizer.synthesize()   → answer text + reasoning chain + confidence
  └─ FollowUpGenerator.generate()     → suggested follow-up questions
```

### Key data structures

- `Chunk` — a chunked piece of a document with tokens, section, page
- `RetrievedChunk` — a `Chunk` with BM25 score and rank
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
```

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

### Adding a new query intent type
1. Add the intent name and trigger phrases to `QueryAnalyzer.QUERY_TYPES`.
2. Add section weights for that intent to `Reranker.SECTION_WEIGHTS`.
3. Add follow-up templates to `FollowUpGenerator.TEMPLATES`.

### Upgrading to embedding-based retrieval
Replace or augment `InvertedIndex.search()`. The reranker, classifier, synthesizer,
and gap detector are all retrieval-agnostic — they only care about the `RetrievedChunk`
interface.

### Upgrading to LLM answer generation
Replace `AnswerSynthesizer._build_answer()`. The reasoning chain construction and
confidence scoring are independent of how the answer text is generated. Pass the
retrieved chunks as context to the LLM and return its response as a list of strings.

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

---

## What Not to Do

- Do not add external dependencies to `core/rag_engine.py`. It must stay stdlib-only.
- Do not raise exceptions inside pipeline stages for low-quality results — return
  a `DecisionOutput` with low confidence instead. Exceptions are for programmer errors only.
- Do not hardcode document IDs or titles anywhere in the engine logic.
- Do not add LLM calls to the core engine without making them clearly optional (e.g.,
  inject a callable or keep behind a flag).
- Do not return raw BM25 scores to the user — they are not interpretable. Always
  normalize to a 0–1 relevance score before surfacing in `EvidenceNode`.

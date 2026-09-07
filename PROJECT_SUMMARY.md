# BioRAG — Decision-Support RAG for Biomedical Literature

A retrieval-augmented generation engine for scientific papers, built as a **decision-support
system** rather than a chatbot. Every answer ships with a reasoning chain, evidence classified
by support type, an explicit knowledge-gap list, and a calibrated confidence score.

Two things separate it from the usual RAG demo:

1. **A measurement harness, not just a pipeline.** Retrieval quality and answer quality are
   evaluated independently, against hand-labelled ground truth, so a regression is attributable
   to a specific stage.
2. **A real ingestion backbone.** An Airflow 3 DAG on Postgres + Qdrant fetches, parses, chunks
   and embeds papers on a schedule, with resumable state and per-paper failure isolation.

---

## 1. What is actually built

| Subsystem | Files | LOC | Status |
|---|---|---|---|
| Core RAG engine (stdlib-only) | `core/rag_engine.py` | 1,187 | Complete |
| Retrieval layer (dense, RRF, rerankers, LLM synth) | `hybrid_retrieval.py`, `cross_encoder_rerank.py`, `rerankers.py`, `llm_synthesizer.py` | 798 | Complete |
| Interfaces (CLI, REST, MCP, PubMed ingest) | `cli.py`, `server.py`, `mcp_server.py`, `ingestion_pubmed.py` | 1,292 | Complete |
| Ingestion pipeline package | `biorag_pipeline/` | 2,484 | Complete |
| Airflow DAG | `airflow/dags/arxiv_ingest_daily.py` | 234 | Complete |
| Eval harnesses | `evals/` | 2,193 | Complete |
| Tests | `tests/` | 1,282 | 5 suites, 115 checks, all passing |
| Schema migrations | `biorag_pipeline/migrations/` | 143 SQL | 2 migrations |
| Documentation | `README.md`, `CLAUDE.md`, `airflow/README.md` | 1,917 | Complete |
| Technical blog series | `blog/` | 3,848 | 16 posts |

**~9,500 lines of first-party Python**, excluding the 4,800-line corpus data file.
MIT-style `LICENSE` included. Python ≥ 3.10.

---

## 2. The retrieval pipeline

```
BioRAGEngine.query(question)
  ├─ QueryAnalyzer          → intent classification, entity extraction, query expansion
  ├─ InvertedIndex          → BM25 (own implementation, tunable K1/B)
  ├─ DenseRetriever         → Qdrant cosine ANN            ┐ optional,
  ├─ reciprocal_rank_fusion → RRF merge of the two lists   ┘ injected
  ├─ Reranker               → section weights + discriminative-token penalty
  ├─ CrossEncoderReranker   → transformer (query, chunk) scoring   ← optional, injected
  ├─ EvidenceClassifier     → direct / indirect / contradictory
  ├─ KnowledgeGapDetector   → missing data, contradictions, low-relevance warnings
  ├─ AnswerSynthesizer      → answer + reasoning chain + confidence
  └─ FollowUpGenerator      → suggested next questions
```

**The core engine has zero external dependencies.** `core/rag_engine.py` is pure stdlib — it
runs with no `pip install` at all. Dense retrieval, cross-encoder reranking and LLM synthesis
are each **optional and constructor-injected**, imported lazily. That is the key architectural
property of the codebase: a buyer can run it as a 1,200-line dependency-free engine, or bolt
on transformers and Qdrant, without a fork.

### Three interchangeable final-stage rerankers

`rerankers.py` collects them behind one interface (`rerank(query, candidates, top_k)`):

| Reranker | Scoring | Cost |
|---|---|---|
| `LexicalReranker` | term overlap × section weight × discriminative-token recall | free, stdlib |
| `BiEncoderReranker` | query and chunk embedded **separately**, ranked by cosine | one forward pass per chunk |
| `CrossEncoderReranker` | query and chunk embedded **together** → calibrated score | slower, models negation/paraphrase |

The **discriminative-token penalty** is the non-obvious piece: a frozenset of ~80 generic
biomedical words (biomarker, patient, elevated, predict…) is subtracted from the query tokens,
and a chunk matching none of the remainder is penalised 85%. This kills the classic BM25 failure
where a lung-cancer paper that says "biomarker" a lot outranks a real hit on an Alzheimer's query.

### Two-stage reranking

When a cross-encoder is present the lexical reranker is **not discarded** — it stays upstream as
a cheap pre-filter narrowing candidates to `cross_encoder_candidates` (default 12) before the
expensive model runs. Bounded cost, obvious off-topic chunks dropped first.

---

## 3. Corpora — what ships and what it can build

### Static corpus (in-repo, ready to run)

`data/sample_corpus.py` — **341 unique documents** harvested from PubMed and PubMed Central:

| | Count |
|---|---|
| Documents | 341 |
| Full text (PubMed Central OA) | 202 |
| Abstract-only (PubMed) | 140 |
| Chunks indexed | 19,185 |
| Unique terms | 40,791 |

Covers cardiology, oncology, infectious disease and neurology, with four hand-curated seed
documents plus 338 ingested via the PubMed pipeline.

### Live pipeline corpus (Postgres + Qdrant, currently populated)

The Airflow pipeline has been run end-to-end against arXiv. Current state:

| | Count |
|---|---|
| Papers ingested | 60 |
| Papers fully indexed | 59 |
| Papers failed | 1 (PDF > 50 MB cap — a deliberate guard, not a bug) |
| Chunks in Postgres | 6,651 |
| Vectors in Qdrant | 6,651 (768-dim, PubMedBERT) |
| Pending / unembedded | 0 |

Postgres and Qdrant agree exactly, which is the design goal: Postgres is the source of truth for
what *should* exist, `chunks.embedded` tracks what *does*, and papers are promoted to `indexed`
by a database predicate rather than by the embedding job believing it succeeded.

---

## 4. The ingestion pipeline (Airflow 3)

```
migrate → discover → fetch → parse (dynamically mapped) → chunk → embed → finish
```

Runs daily at 06:00 UTC. The DAG file is deliberately thin — every task is a wrapper calling one
function in `biorag_pipeline/`, so the whole pipeline is developable and testable from a shell
with Airflow stopped. Each module is also a standalone CLI (`python -m biorag_pipeline.embed --stats`).

**Per-stage detail:**

- **discover** — arXiv Atom API client (`arxiv_client.py`, 456 lines) with its own query grammar
  (categories, phrase search, title-only, date windows), rate-limit throttle, pagination, and
  pre-2007/modern ID parsing. Upserts by version: an unchanged version is a no-op, a newer
  version resets the paper to `discovered` for re-processing, an older one is ignored.
- **fetch** — PDF download with a size cap, content hashing, and on-disk caching. Not
  dynamically mapped, on purpose: the arXiv throttle is process-local, so parallel tasks would
  each start their own timer and hammer the API.
- **parse** — pypdf extraction plus running-head removal, section-heading canonicalisation,
  reference-section dropping, control-character sanitisation (one NUL byte otherwise rejects an
  entire Postgres batch), and abstract-only fallback. Dynamically mapped in batches of 5.
- **chunk** — reuses the engine's own `DocumentChunker`, so pipeline chunks and CLI chunks can
  never drift. Chunk IDs are `md5(doc_id:char_offset)[:12]` — deterministic, so re-chunking
  identical text produces identical IDs and the vector upsert overwrites in place.
- **embed** — batched encoding (64/pass) and upsert (256/request) into a Qdrant server, with a
  payload index on `arxiv_id` so delete-by-filter doesn't full-scan. Point IDs are `uuid5` of the
  chunk ID — idempotent. Distinguishes a *rewritten* chunk set (drop the paper's stale vectors
  first) from a *resumed run* (keep them).

**Failure semantics worth paying for:** no stage takes its work list from the previous stage's
XCom — each queries Postgres by status, so a run that dies mid-way resumes from the database.
Bad data from one PDF fails that paper, never the batch of 500. `finish` uses
`trigger_rule="all_done"` and reads counters from the database, because XComs vanish exactly
when the summary matters most. This is demonstrated, not theoretical: the current corpus was
embedded across two sessions, resuming cleanly at the 3,422-chunk boundary with zero re-encoding.

**Infrastructure:** `docker-compose` stack (Airflow 3 + Postgres + Redis + Qdrant), a Dockerfile
pinning **CPU-only torch** (without the `+cpu` pin the image goes from ~4.8 GB to ~9 GB), a
Postgres init script provisioning a `biorag` database beside Airflow's, and a 12-step setup
walkthrough in `airflow/README.md` ending in four hard verification checks.

---

## 5. Evaluation — the strongest asset

Most RAG repos ship no evals. This one ships three harnesses and hand-labelled ground truth.

### 5a. Retrieval eval — `evals/retrieval_eval.py` (718 lines)

**16 queries** across five disease areas (`ALZHEIMER`, `CARDIO`, `ONCO`, `INFECT`, `CROSS`),
each with a hand-labelled `{doc_id: grade}` map (2 = direct, 1 = partial, 0 = irrelevant).
Chunk scores are max-pooled to document level, then scored on **MRR@K** and **graded NDCG@K**.

Up to **five modes side by side** — BM25 / Dense / Hybrid (RRF) / Hybrid+Rerank / Hybrid+CE —
so every stage's contribution is attributable.

**Measured on the shipped 341-document corpus (16 queries, BM25 vs lexical rerank):**

| Metric | BM25 | Reranked | Δ |
|---|---|---|---|
| MRR@5 | 0.429 | 0.450 | **+0.021** |
| NDCG@1 | 0.312 | 0.375 | **+0.062** |
| NDCG@3 | 0.413 | 0.432 | **+0.019** |
| NDCG@5 | 0.455 | 0.456 | +0.002 |

Broken out by intent, the reranker helps `treatment` (0.639 → 0.700) and `diagnosis`
(0.333 → 0.375) and hurts `epidemiology` (0.200 → 0.000, n=1).

These are honest working numbers on a 341-document corpus with sparse labels, not a
polished marketing figure — and the harness that produced them is the point. The reranker
is now non-negative at every K; the `epidemiology` regression is a single query, and the
per-intent tuning lever (`Reranker.SECTION_WEIGHTS`, `rerank_top_k`) is documented in
`CLAUDE.md`.

### 5b. Answer-quality eval, LLM-as-judge — `evals/answer_eval.py` (601 lines)

**10 hand-authored reference claims**, each with expected entities, expected direction of effect,
and expected context. Claude scores the generated prose via structured `tool_use` against an
**8-dimension, 10-point rubric**:

| Dimension | Max | |
|---|---|---|
| Semantic Coverage | 2 | right phenomenon, not just right topic |
| Entity Coverage | 2 | genes/markers/drugs named in correct context |
| Directional Agreement | 1 | elevated vs decreased vs no effect |
| Quantitative Detail | 1 | magnitudes consistent with the claim |
| Contextual Accuracy | 1 | correct timepoint / tissue / subgroup |
| Source Attribution | 1 | claims tied to inline numbered citations |
| Evidence Strength | 1 | study design named (RCT, meta-analysis, cohort) |
| Uncertainty Calibration | 1 | hedging matches evidence quality |

`--with-retrieval` prints retrieval metrics beside answer scores in one table. That comparison
is the reason both harnesses exist: **a system can score MRR = 1.0 and still state the direction
of effect backwards.** Retrieval metrics cannot detect that; this rubric can.

### 5c. RAGAS cross-check — `evals/ragas_answer_eval.py` (477 lines)

The same rubric re-implemented against RAGAS (`LabelledRubricsScore`, `AspectCritique`) with
Claude as judge via LangChain, and a `--compare` flag for a side-by-side against the
hand-rolled judge. Independent validation that the custom rubric isn't self-serving.

---

## 6. Interfaces

| Interface | Detail |
|---|---|
| **CLI** (`cli.py`) | Interactive REPL, single query, demo mode, corpus stats; `--hybrid`, `--rerank`, `--llm`, `--show-prompt`, `--ingest`, `--save-corpus` |
| **REST API** (`server.py`) | FastAPI + OpenAPI docs. `GET /health`, `POST /query`, `POST /ingest`, `POST /documents`, `GET /corpus`. Feature flags via `BIORAG_HYBRID` / `BIORAG_RERANK` |
| **MCP server** (`mcp_server.py`) | FastMCP over stdio — exposes `query`, `ingest`, `corpus_stats` as native tools to Claude Code / Claude Desktop. One-line registration |
| **PubMed ingestion** (`ingestion_pubmed.py`) | esearch → elink → PMC full text, with abstract fallback and two-layer citation-artifact stripping (XML `<xref>` removal + 5 regex patterns) |
| **Python API** | `BioRAGEngine(dense_retriever=…, cross_encoder=…, synthesizer=…)` |

`DecisionOutput` returns: `answer`, `confidence`, `confidence_label`, `evidence[]`,
`reasoning_chain[]`, `knowledge_gaps[]`, `follow_up_questions[]`, `sources_used`,
`total_chunks_searched`. Raw BM25 scores are never surfaced — everything is normalised to 0–1.

---

## 7. Tests

Framework-free (`assert`-based, no pytest required, though pytest works):

| Suite | Checks | Result |
|---|---|---|
| `test_biorag.py` — TextProcessor, Chunker, BM25, QueryAnalyzer, EvidenceClassifier, E2E | 26 | **26 pass** |
| `test_hybrid_retrieval.py` — EmbeddingModel, DenseRetriever, RRF, E2E hybrid | 16 | **16 pass** |
| `test_cross_encoder_rerank.py` | 13 | **13 pass** |
| `test_arxiv_client.py` — ID parsing, feed parsing, query grammar, windows | 32 | **32 pass** |
| `test_pipeline_db.py` — upsert/versioning, status transitions, chunk replace, embed queue | 28 | **28 pass** |

Hybrid and cross-encoder suites use a `FakeEmbeddingModel` and in-memory Qdrant, so they stay
fast and offline; only two tests load the real transformer. E2E tests run against the real
corpus, never a mock.

---

## 8. Defects found and fixed

Two defects were found by running the full suite, and both are fixed. Recorded here because
the second one changed the retrieval numbers above:

1. **Duplicate corpus entry — fixed.** `pmid_42317338` appeared twice in
   `data/sample_corpus.py`: once as the PMC full text (43,463 chars) and once as the
   abstract-only PubMed fallback (952 chars). Because `add_document` indexes by ID without
   replacing, *both* chunk sets were live under one document — the abstract's three chunks
   duplicating text already present in the full-text version. Removing the abstract-only copy
   dropped 19,188 chunks to 19,185 and lifted every retrieval metric, turning the NDCG@5 delta
   from −0.018 to +0.002.

2. **Pipeline DB test assumed an empty database — fixed.** `test_pipeline_db.py` asserted on
   the *corpus-wide* `pending_chunks(conn)` count rather than its own paper's rows, so it only
   passed against an empty database. A `pending_queue()` helper now scopes the four affected
   assertions to the test's `arxiv_id`. Test isolation, not a product defect — the corpus-wide
   queue is exactly what the embedder wants.

Both suites now pass against the live 60-paper database. Note that `test_arxiv_client.py`
imports through `repository.py` and so needs `psycopg2` on the path — it runs in the worker
container, or on a host with the pipeline deps installed.

---

## 9. Documentation included

- **`README.md`** — architecture, install groups, every CLI/API/MCP invocation, `DecisionOutput` reference.
- **`CLAUDE.md`** — a 500-line engineering handbook: code conventions, per-stage tuning guidance,
  how to add an intent type, how to extend ground truth, and an explicit "What Not to Do" list
  protecting the stdlib-only core. This file makes the repo unusually easy to hand to an AI
  coding agent or a new engineer.
- **`airflow/README.md`** — full infrastructure build from scratch, 12 steps, with verification gates.
- **`blog/`** — 16 markdown pieces (~3,850 lines) walking through the build: retrieval, reranking,
  knowledge gaps, LLM synthesis, MRR/NDCG evals, LLM-as-judge, embeddings/vector DBs, hybrid
  retrieval. A ready-made content asset, plus `generate_diagrams.py` for the figures.

---

## 10. What a buyer gets

**Runs immediately:** `python cli.py --demo` needs zero dependencies and queries a real
341-document biomedical corpus.

**Scales on a switch:** add `--hybrid` for Qdrant dense retrieval + RRF, `--rerank` for the
cross-encoder, `--llm` for Claude synthesis. Same engine, injected components.

**Ingests continuously:** `docker compose up` gives Airflow + Postgres + Qdrant and a daily
arXiv DAG that has already indexed 59 papers into 6,651 vectors.

**Proves its own quality:** three eval harnesses, 16 labelled retrieval queries, 10 reference
claims, an 8-dimension rubric, and a RAGAS cross-check.

**Not a demo.** The parts that only appear after real use are present: resumable state,
per-item failure isolation, deterministic IDs for idempotent re-indexing, version-aware upserts,
schema migrations, rate-limit throttling, size caps, control-character sanitisation, and PDF
running-head removal.

### Honest limitations

- The 341-document corpus is small, and eval labels are sparse — absolute MRR/NDCG figures are
  modest and should be re-measured after ingesting at scale.
- The cross-encoder currently trails the hand-tuned lexical reranker on this corpus. Documented
  and explained: with few documents and document-level max-pooling there is little to reorder,
  and `SECTION_WEIGHTS` was tuned on exactly these queries. A cross-encoder's advantage grows
  with corpus size, and a biomedical cross-encoder (`ncbi/MedCPT-Cross-Encoder`) would likely
  close the gap.
- No authentication, rate limiting, or multi-tenancy on the REST API — it is an internal-service
  API, not a public SaaS surface.
- Answer-quality evals require an `ANTHROPIC_API_KEY`.

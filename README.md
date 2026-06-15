# BioRAG: Decision-Support RAG System

A production-grade **Retrieval-Augmented Generation** engine for scientific and biomedical
documents, designed as a **decision-support system** — not a simple chatbot.

---

## Architecture

```
Query → QueryAnalyzer → InvertedIndex (BM25) ─┐
                     →  DenseRetriever (Qdrant) ┴→ RRF fusion → Reranker → EvidenceClassifier
                       (optional, injected)                                       ↓
                                                                   KnowledgeGapDetector
                                                                               ↓
                                                                   AnswerSynthesizer → DecisionOutput
```

Dense retrieval is **optional and injected** via `BioRAGEngine(dense_retriever=...)`. With no
dense retriever (the default) the pipeline is pure BM25 and the core engine stays stdlib-only;
when one is supplied, BM25 and dense results are fused with Reciprocal Rank Fusion before reranking.

### Core Components

| Component | Description |
|-----------|-------------|
| `TextProcessor` | Biomedical tokenizer with stopword filtering and abbreviation expansion (DNA, PCR, ELISA…) |
| `DocumentChunker` | Overlapping sliding-window chunker with section detection (Abstract/Methods/Results…) |
| `InvertedIndex` | BM25 (Okapi) inverted index — outperforms TF-IDF for biomedical text |
| `DenseRetriever` | Optional semantic retrieval over a Qdrant collection of `S-PubMedBert` embeddings; fused with BM25 via Reciprocal Rank Fusion (`hybrid_retrieval.py`) |
| `QueryAnalyzer` | Intent classification (mechanism/comparison/treatment/diagnosis/prognosis…) + entity extraction |
| `Reranker` | Section-aware reranker with discriminative-token recall penalty to suppress off-topic false positives |
| `EvidenceClassifier` | Labels each chunk as `direct`, `indirect`, or `contradictory` evidence |
| `KnowledgeGapDetector` | Detects missing quantitative data, contradictions, low-relevance matches |
| `AnswerSynthesizer` | Builds structured answer with explicit reasoning chain and confidence scoring |
| `ClaudeSynthesizer` | Optional LLM-backed synthesizer — replaces rule-based answer with a Claude API call, strictly grounded in retrieved evidence |
| `FollowUpGenerator` | Generates intent-aware follow-up questions |

---

## Installation

```bash
# Core engine: zero dependencies (stdlib only)
python cli.py --demo

# For PubMed/PMC ingestion:
pip install requests

# For the REST API:
pip install fastapi uvicorn pydantic
python server.py  # http://localhost:8000

# For the MCP server (Claude Code / Claude Desktop integration):
pip install "mcp[cli]"

# For LLM answer synthesis (ClaudeSynthesizer):
pip install anthropic

# For hybrid retrieval (BM25 + dense via Qdrant):
pip install qdrant-client sentence-transformers
python cli.py --hybrid --query "What plasma proteins predict Alzheimer's?"
```

---

## Quick Start

### CLI

```bash
# Interactive REPL
python cli.py

# Single query
python cli.py --query "What biomarkers predict CVD risk in T2DM?"

# Run all demo queries
python cli.py --demo

# Corpus stats
python cli.py --stats

# Ingest from PubMed/PMC then enter interactive mode (default 10 papers)
python cli.py --ingest "alzheimer's disease biomarkers"

# Fetch more papers
python cli.py --ingest "alzheimer's disease biomarkers" --ingest-max 50

# Ingest and persist to data/sample_corpus.py (survives process restart)
python cli.py --ingest "alzheimer's disease biomarkers" --ingest-max 25 --save-corpus

# Use Claude as the answer synthesizer (requires: pip install anthropic)
python cli.py --llm --query "What biomarkers predict Alzheimer's disease?"

# Use Claude and print the full prompt sent for each answer
python cli.py --llm --show-prompt --query "What biomarkers predict Alzheimer's disease?"

# LLM mode in interactive REPL
python cli.py --llm --show-prompt

# Hybrid retrieval: BM25 + dense (Qdrant) fused via RRF (requires: pip install qdrant-client sentence-transformers)
python cli.py --hybrid --query "What plasma proteins predict Alzheimer's?"

# Hybrid + Claude synthesis
python cli.py --hybrid --llm --query "What plasma proteins predict Alzheimer's?"

# Ingest into a hybrid index (chunks flow to both BM25 and Qdrant)
python cli.py --hybrid --ingest "alzheimer's disease biomarkers" --ingest-max 25
```

Inside the interactive REPL you can also ingest on the fly, with an optional paper count:

```
Query> ingest CRISPR cancer therapy
Query> ingest CRISPR cancer therapy 25
```

### Python API

```python
from core.rag_engine import BioRAGEngine

# Default: rule-based synthesizer, zero external dependencies
engine = BioRAGEngine(
    chunk_size=512,
    chunk_overlap=64,
    retrieval_top_k=15,
    rerank_top_k=5,
)

# Optional: Claude-backed synthesizer (requires: pip install anthropic)
from llm_synthesizer import ClaudeSynthesizer
engine = BioRAGEngine(synthesizer=ClaudeSynthesizer())

# Optional: hybrid BM25 + dense retrieval (requires: pip install qdrant-client sentence-transformers)
from hybrid_retrieval import EmbeddingModel, DenseRetriever
engine = BioRAGEngine(dense_retriever=DenseRetriever(EmbeddingModel()))


# Add documents
engine.add_document(
    doc_id="paper_001",
    title="My Research Paper",
    text="Abstract\n\nThis study examines...",
    metadata={"year": 2023, "journal": "Nature"}
)

# Query
result = engine.query("What are the main findings on cardiovascular biomarkers?")

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence_label} ({result.confidence:.0%})")
print(f"Evidence nodes: {len(result.evidence)}")
print(f"Knowledge gaps: {result.knowledge_gaps}")
```

### REST API

```bash
python server.py
# → http://localhost:8000

# Enable hybrid BM25 + dense retrieval (no flag needed — toggled by env var)
BIORAG_HYBRID=1 python server.py

# POST /query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does PD-L1 expression predict immunotherapy response?"}'

# POST /ingest  (fetch PubMed/PMC papers and add to running corpus)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"query": "alzheimer biomarkers", "max_results": 10}'

# POST /documents  (add a single document manually)
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "new_001", "title": "My Paper", "text": "..."}'

# GET /corpus
curl http://localhost:8000/corpus
```

### MCP Server (Claude Code / Claude Desktop)

The MCP server exposes the BioRAG engine as three tools that Claude can call
directly inside a conversation.

**Register once:**

```bash
pip install "mcp[cli]"
claude mcp add biorag python /absolute/path/to/mcp_server.py
```

Or add to `.claude/settings.json` for project-scoped registration:

```json
{
  "mcpServers": {
    "biorag": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

**Verify:**

```bash
claude mcp list          # shows: biorag ✓ Connected
claude mcp get biorag    # shows tool definitions
```

**Available tools:**

| Tool | Description |
|---|---|
| `query(question)` | Query the RAG engine — returns answer, confidence, evidence, knowledge gaps |
| `ingest(pubmed_query, max_results)` | Fetch PubMed/PMC papers and add to corpus |
| `corpus_stats()` | Return document/chunk/term counts for the running corpus |

**Usage inside Claude Code (after restarting the session):**

```
Use biorag to query: what biomarkers predict Alzheimer's disease?
Use biorag to ingest 20 papers on CRISPR cancer therapy
Use biorag corpus_stats to show the current corpus
```

The server loads the sample corpus on startup. Documents ingested via `ingest`
persist for the lifetime of the server process.

---

## DecisionOutput Structure

```python
@dataclass
class DecisionOutput:
    query: str                          # Original question
    answer: str                         # Synthesized answer with source citations
    confidence: float                   # 0.0–1.0 calibrated confidence score
    confidence_label: str               # "High" | "Moderate" | "Low" | "Insufficient"
    evidence: list[EvidenceNode]        # Ranked evidence with support type
    reasoning_chain: list[ReasoningStep] # Explicit step-by-step reasoning trace
    knowledge_gaps: list[str]           # What the corpus doesn't cover
    follow_up_questions: list[str]      # Suggested next queries
    sources_used: int
    total_chunks_searched: int
```

### Evidence Support Types

- **direct** — chunk contains statistical results, significance statements, or direct confirmations
- **indirect** — chunk is relevant but does not directly confirm the query
- **contradictory** — chunk contains language suggesting refutation or failure

---

## Tests

```bash
python tests/test_biorag.py
# 26 tests · TextProcessor · Chunker · BM25 Index · QueryAnalyzer · EvidenceClassifier · E2E Pipeline

python tests/test_hybrid_retrieval.py
# 16 tests · EmbeddingModel · DenseRetriever · reciprocal_rank_fusion · E2E hybrid pipeline
```

The hybrid suite uses a fake embedding model and an in-memory Qdrant collection so most tests
stay fast and offline; only the two `EmbeddingModel` tests load the real sentence-transformers
model (once) to verify embedding dimension and caching.

---

## Retrieval Evals

The `evals/` directory contains a retrieval quality harness that measures **MRR** (Mean
Reciprocal Rank) and **NDCG@K** independently for the BM25 and reranker stages.

```bash
# Full eval set (16 queries across all disease areas)
python evals/retrieval_eval.py

# Alzheimer's disease subset only (7 queries, Q01–Q07)
python evals/retrieval_eval.py --alzheimer-only

# Show per-query detail: top-3 retrieved docs vs ground truth
python evals/retrieval_eval.py --verbose

# Custom K values
python evals/retrieval_eval.py --ks 1 5 10

# Four-mode comparison: BM25 / Dense / Hybrid (RRF) / Hybrid+Rerank
python evals/retrieval_eval.py --hybrid --alzheimer-only --verbose
```

Sample output:

```
  Metric           BM25   Reranked        Δ
  ─────────────────────────────────────────
  MRR@5           1.000      1.000   +0.000
  NDCG@1          0.958      0.958   +0.000
  NDCG@3          0.978      0.970   -0.008
  NDCG@5          0.985      0.970   -0.015
```

With `--hybrid`, all four retrieval modes are compared side by side so each stage's
contribution is attributable (requires `qdrant-client` + `sentence-transformers`):

```
  Metric                BM25           Dense          Hybrid   Hybrid+Rerank
  ──────────────────────────────────────────────────────────────────────────
  MRR@5                0.714           0.719           0.695           0.821
  NDCG@1               0.571           0.571           0.571           0.714
  NDCG@3               0.752           0.733           0.714           0.804
  NDCG@5               0.752           0.788           0.770           0.866
```

**Metrics explained:**

- **MRR@K** — Mean Reciprocal Rank: measures where the first relevant document lands.
  High MRR means the system surfaces the right answer quickly.
- **NDCG@K** — Normalized Discounted Cumulative Gain: rewards the full ranked list using
  graded relevance (2 = directly answers, 1 = partially relevant, 0 = irrelevant).
  Penalises a system that buries a highly-relevant document at rank 3 vs rank 1.

The Δ column shows `Reranked − BM25`. A negative NDCG@3 delta indicates the reranker's
`rerank_top_k` budget is trimming documents that are partially relevant to multi-document
queries — a useful signal when tuning the reranker.

**Ground truth** is in `evals/ground_truth.py` as `EVAL_QUERIES` — a list of
`RetrievalQuery` objects with hand-labelled `{doc_id: grade}` relevance dicts. Add new
queries there after ingesting new papers to keep the eval set growing with the corpus.

---

## Answer Quality Evals (LLM-as-Judge)

While the retrieval eval measures *whether the right document was found*, the answer
quality eval measures *whether the answer said the right thing* — a separate failure mode
that retrieval metrics cannot detect.

`evals/answer_eval.py` runs `BioRAGEngine.query()` for each eval query and passes the
answer to Claude as an independent judge, scoring it on an eight-dimension rubric (max 10
points per query):

| Dimension | Max | What it checks |
|---|---|---|
| Semantic Coverage | 2 | Same specific biological conclusion, not just the same topic area |
| Entity Coverage | 2 | All expected genes/markers/drugs named in the correct context |
| Directional Agreement | 1 | Direction of effect (elevated/decreased/no effect) explicitly stated and correct |
| Quantitative Detail | 1 | At least one magnitude or statistic consistent with the reference |
| Contextual Accuracy | 1 | Finding placed in the claim-specific context (timepoint, tissue, subgroup) |
| Source Attribution | 1 | Every factual claim accompanied by inline citations ([1], [2], etc.) |
| Evidence Strength | 1 | Study design or evidence type explicitly named (RCT, meta-analysis, cohort, in vitro) |
| Uncertainty Calibration | 1 | Expressed confidence matches evidence quality — assertive for strong evidence, hedged for weak |

```bash
# Score all 10 reference claims (requires ANTHROPIC_API_KEY)
python evals/answer_eval.py

# Alzheimer's subset only (4 claims)
python evals/answer_eval.py --alzheimer-only

# Full answer + judge rationale per query
python evals/answer_eval.py --verbose

# Use ClaudeSynthesizer for answers
python evals/answer_eval.py --llm

# Combined table: retrieval MRR/NDCG alongside answer quality scores
python evals/answer_eval.py --with-retrieval
```

Sample output (LLM synthesizer):

```
  Dimension                    Mean   Max  Bar
  ──────────────────────────────────────────────────
  Semantic Coverage            1.40  /2    ████░
  Entity Coverage              1.70  /2    ████░
  Directional Agreement        0.80  /1    ████░
  Quantitative Detail          0.30  /1    ██░░░
  Contextual Accuracy          1.00  /1    █████
  Source Attribution           1.00  /1    █████
  Evidence Strength            0.40  /1    ██░░░
  Uncertainty Calibration      0.90  /1    ████░
  ──────────────────────────────────────────────────
  Total Score                  7.50  /10
```

**Ground truth** is in `evals/answer_ground_truth.py` as `ANSWER_CLAIMS` — 10
`AnswerClaim` objects with `reference_claim`, `expected_entities`, `expected_direction`,
and `expected_context` fields. Add new claims alongside new retrieval queries to keep
both eval sets growing with the corpus.

---

## Key Design Decisions

### Why BM25 as the default, with dense retrieval optional?
BM25 requires zero external dependencies and zero API calls. For domain-specific biomedical
text with precise terminology (gene names, drug names, statistical notation), BM25 matches
exact terms reliably. Dense embeddings offer semantic similarity but pull in `torch`/`qdrant`
and can hallucinate relevance for rare biomedical entities. So BM25 is the default and the core
engine stays stdlib-only, while dense retrieval is an **opt-in upgrade** injected via
`dense_retriever` — you get exact-match precision for free and semantic recall when you want it.

### Why hybrid + Reciprocal Rank Fusion?
The two retrievers fail in complementary ways: BM25 misses paraphrases ("plasma proteins" vs
"blood biomarkers"), dense misses rare exact tokens (specific gene IDs). RRF
(`score = Σ 1/(k + rank_i)`, `k=60`) combines them using only rank position — no score
normalisation between incompatible scales (BM25 magnitudes vs cosine ∈ [-1, 1]). The fused
score is written back to `RetrievedChunk.score`, so the reranker and every downstream stage are
untouched. In the eval above, Hybrid+Rerank beats BM25 alone on every metric.

### Why explicit reasoning chains?
Clinical decision support systems must be auditable. An opaque answer is dangerous in
biomedical contexts. Every conclusion is traceable to: (1) query interpretation, (2)
evidence retrieved, (3) evidence classified, (4) gaps identified.

### Why knowledge gap detection?
Overconfident AI answers in clinical contexts can cause harm. The gap detector explicitly
flags when: quantitative data is absent, contradictions exist, relevance is low, or query
entities have no coverage — surfacing uncertainty rather than hiding it.

### Why discriminative-token recall in the Reranker?
BM25 only rewards term presence — it never penalises term absence. A lung cancer paper
that heavily uses "biomarker" and "disease" can score near the top for an Alzheimer's
query even though it never mentions "alzheimer". The `Reranker` maintains a
`_GENERIC_BIO_TERMS` set of ~80 words that appear in virtually every biomedical paper
(biomarker, disease, patient, predict, detect, assess…). Query tokens not in this set
are treated as **discriminative**; a chunk matching none of them receives an 85% score
penalty. This keeps false positives out of the top-5 evidence nodes without any changes
to BM25 or the index.

### Why inject the synthesizer rather than subclass BioRAGEngine?
`BioRAGEngine.__init__` accepts an optional `synthesizer` argument. Passing
`ClaudeSynthesizer()` swaps in LLM answer generation while leaving every other pipeline
stage (retrieval, reranking, classification, confidence scoring) unchanged. The default
`AnswerSynthesizer` remains pure stdlib — no API key required for the core engine.

---

## Extending

### Hybrid dense retrieval (shipped, opt-in)
`hybrid_retrieval.py` provides `EmbeddingModel`, `DenseRetriever`, and `reciprocal_rank_fusion`.
Inject a `DenseRetriever` and the engine fuses BM25 + dense results before reranking — every
other stage (reranker, classifier, synthesizer, gap detector) is unchanged.

```python
from hybrid_retrieval import EmbeddingModel, DenseRetriever
from core.rag_engine import BioRAGEngine

# File-based Qdrant (default): vectors persist in ./qdrant_data and survive restarts.
# add_chunks() dedupes by chunk id, so the corpus is embedded once, not on every launch.
engine = BioRAGEngine(dense_retriever=DenseRetriever(EmbeddingModel()))
result = engine.query("What plasma biomarkers predict Alzheimer's disease?")

# Swap the embedding model, or use an in-memory collection for tests/evals:
dr = DenseRetriever(EmbeddingModel("BAAI/bge-small-en-v1.5"), qdrant_path=":memory:")
```

Default model: `pritamdeka/S-PubMedBert-MS-MARCO` (768-dim, PubMed-tuned). RRF uses `k=60`;
tune it via `reciprocal_rank_fusion(..., rrf_k=...)`.

### Using LLM answer generation
`ClaudeSynthesizer` in `llm_synthesizer.py` is a drop-in replacement for
`AnswerSynthesizer`. It calls `claude-sonnet-4-6` at `temperature=0` with the retrieved
evidence formatted as numbered excerpts, and a strict system prompt that forbids drawing
on training knowledge. The last prompt sent is stored in `synthesizer.last_prompt` for
inspection.

```python
from llm_synthesizer import ClaudeSynthesizer, SYSTEM_PROMPT
from core.rag_engine import BioRAGEngine

engine = BioRAGEngine(synthesizer=ClaudeSynthesizer())
result = engine.query("What plasma biomarkers predict Alzheimer's disease?")

# Inspect what was sent to Claude
print(engine.synthesizer.last_prompt["system"])
print(engine.synthesizer.last_prompt["user"])
```

To extend or replace the synthesizer, subclass `AnswerSynthesizer` and override
`synthesize()`. The reasoning chain steps and confidence scoring are inherited.

---

## File Structure

```
biorag/
├── core/
│   └── rag_engine.py       # All core components (single file, zero deps)
├── data/
│   └── sample_corpus.py    # Biomedical corpus sourced from PubMed Central (citation-cleaned)
├── evals/
│   ├── ground_truth.py        # 16 RetrievalQuery objects with doc-level relevance grades (0/1/2)
│   ├── retrieval_eval.py      # MRR and NDCG@K harness — evaluates BM25 vs reranker separately
│   ├── answer_ground_truth.py # 10 AnswerClaim objects with reference claims and rubric targets
│   └── answer_eval.py         # LLM-as-judge harness — 8-dimension rubric scored by Claude
├── tests/
│   ├── test_biorag.py            # 26 unit + integration tests
│   └── test_hybrid_retrieval.py  # 16 tests for EmbeddingModel / DenseRetriever / RRF
├── server.py               # FastAPI REST server (includes /ingest endpoint; BIORAG_HYBRID env var)
├── cli.py                  # Interactive terminal interface (--llm, --show-prompt, --save-corpus, --hybrid)
├── llm_synthesizer.py      # ClaudeSynthesizer — LLM-backed answer synthesis, strictly grounded
├── hybrid_retrieval.py     # EmbeddingModel + DenseRetriever (Qdrant) + reciprocal_rank_fusion
├── ingestion_pubmed.py     # PubMed/PMC ingestion pipeline with save_to_corpus()
├── mcp_server.py           # MCP server — exposes query/ingest/corpus_stats tools
├── requirements.txt
└── README.md
```

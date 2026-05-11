# BioRAG: Decision-Support RAG System

A production-grade **Retrieval-Augmented Generation** engine for scientific and biomedical
documents, designed as a **decision-support system** — not a simple chatbot.

---

## Architecture

```
Query → QueryAnalyzer → InvertedIndex (BM25) → Reranker → EvidenceClassifier
                                                              ↓
                                               KnowledgeGapDetector
                                                              ↓
                                               AnswerSynthesizer → DecisionOutput
```

### Core Components

| Component | Description |
|-----------|-------------|
| `TextProcessor` | Biomedical tokenizer with stopword filtering and abbreviation expansion (DNA, PCR, ELISA…) |
| `DocumentChunker` | Overlapping sliding-window chunker with section detection (Abstract/Methods/Results…) |
| `InvertedIndex` | BM25 (Okapi) inverted index — outperforms TF-IDF for biomedical text |
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
```

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

## Key Design Decisions

### Why BM25, not dense embeddings?
BM25 requires zero external dependencies and zero API calls. For domain-specific biomedical
text with precise terminology (gene names, drug names, statistical notation), BM25 matches
exact terms reliably. Dense embeddings offer semantic similarity but require GPU/API access
and can hallucinate relevance for rare biomedical entities.

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

### Adding dense retrieval (optional upgrade path)
```python
# Drop-in replacement for InvertedIndex.search():
# Use sentence-transformers or OpenAI embeddings + FAISS/Qdrant
# The rest of the pipeline (reranker, classifier, synthesizer) is embedding-agnostic
from sentence_transformers import SentenceTransformer
import faiss
```

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
│   ├── ground_truth.py     # 16 annotated queries with doc-level relevance grades (0/1/2)
│   └── retrieval_eval.py   # MRR and NDCG@K harness — evaluates BM25 vs reranker separately
├── tests/
│   └── test_biorag.py      # 26 unit + integration tests
├── server.py               # FastAPI REST server (includes /ingest endpoint)
├── cli.py                  # Interactive terminal interface (--llm, --show-prompt, --save-corpus)
├── llm_synthesizer.py      # ClaudeSynthesizer — LLM-backed answer synthesis, strictly grounded
├── ingestion_pubmed.py     # PubMed/PMC ingestion pipeline with save_to_corpus()
├── mcp_server.py           # MCP server — exposes query/ingest/corpus_stats tools
├── requirements.txt
└── README.md
```

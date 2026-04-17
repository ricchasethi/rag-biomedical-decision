# BioRAG — Decision-Support RAG System

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
| `Reranker` | Section-aware reranker: prioritizes Results/Methods for factual queries |
| `EvidenceClassifier` | Labels each chunk as `direct`, `indirect`, or `contradictory` evidence |
| `KnowledgeGapDetector` | Detects missing quantitative data, contradictions, low-relevance matches |
| `AnswerSynthesizer` | Builds structured answer with explicit reasoning chain and confidence scoring |
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
```

Inside the interactive REPL you can also ingest on the fly, with an optional paper count:

```
Query> ingest CRISPR cancer therapy
Query> ingest CRISPR cancer therapy 25
```

### Python API

```python
from core.rag_engine import BioRAGEngine

engine = BioRAGEngine(
    chunk_size=512,
    chunk_overlap=64,
    retrieval_top_k=15,
    rerank_top_k=5,
)

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

### Adding LLM answer generation
```python
# Replace AnswerSynthesizer._build_answer() with an LLM call:
# Pass retrieved chunks as context, use the existing reasoning chain as the prompt
import anthropic
client = anthropic.Anthropic()
```

---

## File Structure

```
biorag/
├── core/
│   └── rag_engine.py       # All core components (single file, zero deps)
├── data/
│   └── sample_corpus.py    # Biomedical corpus sourced from PubMed Central
├── tests/
│   └── test_biorag.py      # 26 unit + integration tests
├── server.py               # FastAPI REST server (includes /ingest endpoint)
├── cli.py                  # Interactive terminal interface (includes ingest command)
├── ingestion_pubmed.py     # PubMed/PMC full-text ingestion pipeline
├── mcp_server.py           # MCP server — exposes query/ingest/corpus_stats tools
├── requirements.txt
└── README.md
```

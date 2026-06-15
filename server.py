"""
BioRAG API Server
━━━━━━━━━━━━━━━━
FastAPI server exposing the RAG decision-support engine.
Run with: python server.py (or uvicorn server:app --reload)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.rag_engine import BioRAGEngine, DecisionOutput
from data.sample_corpus import SAMPLE_DOCUMENTS
from ingestion_pubmed import ingest_pubmed, IngestionError

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BioRAG Decision-Support API",
    description="RAG-based decision support for biomedical/scientific documents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Engine ────────────────────────────────────────────────────────────

# Toggle hybrid BM25 + dense retrieval via env var so the process needs no flag.
dense_retriever = None
if os.getenv("BIORAG_HYBRID"):
    from hybrid_retrieval import EmbeddingModel, DenseRetriever
    dense_retriever = DenseRetriever(EmbeddingModel())

engine = BioRAGEngine(
    chunk_size=480,
    chunk_overlap=60,
    retrieval_top_k=12,
    rerank_top_k=5,
    dense_retriever=dense_retriever,
)

# Preload sample corpus
_corpus_loaded = False

def load_corpus():
    global _corpus_loaded
    if _corpus_loaded:
        return
    for doc in SAMPLE_DOCUMENTS:
        engine.add_document(
            doc_id=doc["id"],
            title=doc["title"],
            text=doc["text"],
            metadata=doc.get("metadata", {}),
        )
    _corpus_loaded = True

# ─── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    max_evidence: Optional[int] = 5


class AddDocumentRequest(BaseModel):
    doc_id: str
    title: str
    text: str
    metadata: Optional[dict] = None


class IngestRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10


class EvidenceNodeOut(BaseModel):
    source_id: str
    doc_title: str
    section: str
    excerpt: str
    relevance_score: float
    support_type: str
    key_terms: list[str]


class ReasoningStepOut(BaseModel):
    step_number: int
    label: str
    content: str
    confidence: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    confidence_label: str
    evidence: list[EvidenceNodeOut]
    reasoning_chain: list[ReasoningStepOut]
    knowledge_gaps: list[str]
    follow_up_questions: list[str]
    sources_used: int
    total_chunks_searched: int
    latency_ms: float


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    load_corpus()


@app.get("/health")
def health():
    load_corpus()
    stats = engine.get_corpus_stats()
    return {
        "status": "ok",
        "corpus": stats,
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    load_corpus()

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    t0 = time.perf_counter()
    result: DecisionOutput = engine.query(req.question)
    latency = (time.perf_counter() - t0) * 1000

    # Trim evidence to requested max
    evidence = result.evidence[:req.max_evidence]

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        confidence=result.confidence,
        confidence_label=result.confidence_label,
        evidence=[EvidenceNodeOut(**{
            "source_id": e.source_id,
            "doc_title": e.doc_title,
            "section": e.section,
            "excerpt": e.excerpt,
            "relevance_score": e.relevance_score,
            "support_type": e.support_type,
            "key_terms": e.key_terms,
        }) for e in evidence],
        reasoning_chain=[ReasoningStepOut(**{
            "step_number": s.step_number,
            "label": s.label,
            "content": s.content,
            "confidence": s.confidence,
        }) for s in result.reasoning_chain],
        knowledge_gaps=result.knowledge_gaps,
        follow_up_questions=result.follow_up_questions,
        sources_used=result.sources_used,
        total_chunks_searched=result.total_chunks_searched,
        latency_ms=round(latency, 1),
    )


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    """Fetch PubMed/PMC papers for *query* and add them to the running corpus.

    Full-text articles are retrieved from PubMed Central where available;
    abstracts are used as a fallback for papers not in PMC.
    Note: this is a synchronous call — for large ``max_results`` values it
    may take several seconds due to NCBI rate-limiting.
    """
    load_corpus()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    docs_before = len(engine.documents)
    try:
        ingest_result = ingest_pubmed(req.query, max_results=req.max_results, engine=engine)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"PubMed/PMC fetch failed: {exc}") from exc

    stats = engine.get_corpus_stats()
    return {
        "status": "ok",
        "query": req.query,
        "documents_added": len(engine.documents) - docs_before,
        "corpus_stats": stats,
        "errors": [
            {"stage": e.stage, "identifier": e.identifier, "reason": e.reason}
            for e in ingest_result.errors
        ],
    }


@app.post("/documents")
def add_document(req: AddDocumentRequest):
    load_corpus()
    n_chunks = engine.add_document(
        doc_id=req.doc_id,
        title=req.title,
        text=req.text,
        metadata=req.metadata,
    )
    return {
        "status": "indexed",
        "doc_id": req.doc_id,
        "chunks_created": n_chunks,
        "corpus_stats": engine.get_corpus_stats(),
    }


@app.get("/corpus")
def get_corpus():
    load_corpus()
    return engine.get_corpus_stats()


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

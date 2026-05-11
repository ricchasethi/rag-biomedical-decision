# mcp_server.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import threading
import time as _time
import uuid

from mcp.server.fastmcp import FastMCP
from core.rag_engine import BioRAGEngine
from data.sample_corpus import SAMPLE_DOCUMENTS
from ingestion_pubmed import ingest_pubmed, IngestionError

mcp = FastMCP("biorag")

# ── Engine ─────────────────────────────────────────────────────────────────────

_engine = BioRAGEngine()
for doc in SAMPLE_DOCUMENTS:
    _engine.add_document(doc["id"], doc["title"], doc["text"],
                         doc.get("metadata"))

# BioRAGEngine carries its own threading.Lock (_engine._lock) that serialises
# concurrent add_document / query / get_corpus_stats calls, so no external
# lock is needed here.

# ── Job registry ───────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _ingest_job(job_id: str, pubmed_query: str, max_results: int) -> None:
    """Worker target: run ingest_pubmed and record the outcome in _jobs."""
    try:
        result = ingest_pubmed(pubmed_query, max_results=max_results, engine=_engine)
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "done",
                "indexed": result.indexed,
                "errors": [
                    {"stage": e.stage, "identifier": e.identifier, "reason": e.reason}
                    for e in result.errors
                ],
                "finished_at": _time.monotonic(),
            })
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "error",
                "error": str(exc),
                "finished_at": _time.monotonic(),
            })


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def query(question: str) -> dict:
    """Query the BioRAG engine with a biomedical question.
    Returns a structured answer with confidence, evidence, and knowledge gaps.
    """
    result = _engine.query(question)
    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "confidence_label": result.confidence_label,
        "evidence": [
            {
                "doc_title": e.doc_title,
                "section": e.section,
                "excerpt": e.excerpt,
                "relevance_score": e.relevance_score,
                "support_type": e.support_type,
            }
            for e in result.evidence
        ],
        "knowledge_gaps": result.knowledge_gaps,
        "follow_up_questions": result.follow_up_questions,
        "sources_used": result.sources_used,
    }


@mcp.tool()
def ingest(pubmed_query: str, max_results: int = 10) -> dict:
    """Start a background PubMed/PMC ingestion job and return its job_id immediately.

    Full text is retrieved from PMC where available; abstracts are used as
    fallback. Call ingest_status(job_id) to poll for completion.
    """
    job_id = uuid.uuid4().hex[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "query": pubmed_query,
            "max_results": max_results,
            "indexed": 0,
            "errors": [],
            "error": "",
            "started_at": _time.monotonic(),
            "finished_at": None,
        }
    threading.Thread(
        target=_ingest_job,
        args=(job_id, pubmed_query, max_results),
        daemon=True,
        name=f"ingest-{job_id}",
    ).start()
    return {
        "job_id": job_id,
        "status": "running",
        "message": (
            f"Ingesting up to {max_results} papers for '{pubmed_query}' in the background. "
            f"Call ingest_status('{job_id}') to poll for results."
        ),
    }


@mcp.tool()
def ingest_status(job_id: str) -> dict:
    """Return the status of a background ingestion job started by ingest().

    When status is 'done', corpus_stats from the updated engine are included.
    When status is 'error', the 'error' field holds the exception message.
    When status is 'running', elapsed_seconds shows how long it has been active.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return {"error": f"Unknown job_id '{job_id}'"}

    result = dict(job)
    now = _time.monotonic()
    started = result.pop("started_at")
    finished = result.pop("finished_at")
    result["elapsed_seconds"] = round((finished if finished else now) - started, 1)

    if result["status"] == "done":
        result["corpus_stats"] = _engine.get_corpus_stats()

    return result


@mcp.tool()
def corpus_stats() -> dict:
    """Return statistics about the currently indexed document corpus."""
    return _engine.get_corpus_stats()


if __name__ == "__main__":
    mcp.run()

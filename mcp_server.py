# mcp_server.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from core.rag_engine import BioRAGEngine
from data.sample_corpus import SAMPLE_DOCUMENTS
from ingestion_pubmed import ingest_pubmed

mcp = FastMCP("biorag")

# Shared engine — loaded once at startup
_engine = BioRAGEngine()
for doc in SAMPLE_DOCUMENTS:
    _engine.add_document(doc["id"], doc["title"], doc["text"], 
                         doc.get("metadata"))


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
    """Fetch papers from PubMed/PMC and add them to the running corpus.
    Full text is retrieved from PMC where available; abstracts are used as
    fallback.
    """
    ingest_pubmed(pubmed_query, max_results=max_results, engine=_engine)
    return _engine.get_corpus_stats()


@mcp.tool()
def corpus_stats() -> dict:
    """Return statistics about the currently indexed document corpus."""
    return _engine.get_corpus_stats()


if __name__ == "__main__":
    mcp.run()
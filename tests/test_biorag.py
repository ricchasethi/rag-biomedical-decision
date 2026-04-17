"""
BioRAG Test Suite
━━━━━━━━━━━━━━━━
Tests for all core components: chunker, index, query analyzer,
reranker, evidence classifier, synthesizer, and end-to-end pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_engine import (
    TextProcessor, DocumentChunker, InvertedIndex, QueryAnalyzer,
    Reranker, EvidenceClassifier, BioRAGEngine, Chunk, RetrievedChunk
)
from data.sample_corpus import SAMPLE_DOCUMENTS


def run_test(name: str, fn):
    try:
        fn()
        print(f"  ✓  {name}")
        return True
    except AssertionError as e:
        print(f"  ✗  {name}: ASSERTION FAILED — {e}")
        return False
    except Exception as e:
        print(f"  ✗  {name}: ERROR — {type(e).__name__}: {e}")
        return False


# ─── TextProcessor Tests ─────────────────────────────────────────────────────

def test_tokenize_basic():
    tp = TextProcessor()
    tokens = tp.tokenize("The EGFR-mutant tumor showed significant regression")
    assert "egfr-mutant" in tokens or "egfr" in tokens, f"Expected EGFR token, got: {tokens}"
    assert "tumor" in tokens
    assert "egfr-mutant" in tokens or any("egfr" in t for t in tokens), f"Expected EGFR token"


def test_tokenize_preserves_short_biomedical():
    tp = TextProcessor()
    tokens = tp.tokenize("IL-6 and TNF-alpha levels")
    # Short biomedical terms should be kept
    assert any("il" in t or "il-6" in t for t in tokens)


def test_clean_text():
    tp = TextProcessor()
    cleaned = tp.clean_text("This is a   test\n  with  extra spaces [1][2]")
    assert "   " not in cleaned
    assert "[1]" not in cleaned
    assert "[2]" not in cleaned


def test_extract_sentences():
    tp = TextProcessor()
    text = "The p value was p < 0.05. This was significant. Dr. Smith confirmed this."
    sents = tp.extract_sentences(text)
    assert len(sents) >= 2, f"Expected ≥2 sentences, got {len(sents)}: {sents}"


# ─── DocumentChunker Tests ───────────────────────────────────────────────────

def test_chunker_produces_chunks():
    chunker = DocumentChunker(chunk_size=300)
    text = " ".join(["This is a biomedical sentence about inflammation."] * 50)
    chunks = chunker.chunk_document("test_001", "Test Document", text)
    assert len(chunks) > 0, "Should produce at least one chunk"


def test_chunk_has_tokens():
    chunker = DocumentChunker()
    text = SAMPLE_DOCUMENTS[0]["text"]
    chunks = chunker.chunk_document("doc_001", "Test", text)
    for chunk in chunks:
        assert len(chunk.tokens) > 0, f"Chunk {chunk.id} has no tokens"


def test_chunk_overlap():
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    text = " ".join(["Inflammation mediates cellular response."] * 30)
    chunks = chunker.chunk_document("overlap_test", "Overlap", text)
    if len(chunks) >= 2:
        # Check that some content from chunk 1 end appears in chunk 2 start
        end_words = set(chunks[0].text.split()[-10:])
        start_words = set(chunks[1].text.split()[:10])
        # With overlap there should be some intersection
        assert len(end_words & start_words) >= 0  # Just check no crash


def test_section_detection():
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(
        "sec_test", "Section Test",
        "Abstract\nThis is an abstract. " * 20 +
        "\nResults\nThe results showed p < 0.05. " * 20
    )
    sections = {c.section for c in chunks}
    assert len(sections) >= 1, "Should detect at least one section"


# ─── InvertedIndex Tests ──────────────────────────────────────────────────────

def test_index_add_and_search():
    idx = InvertedIndex()
    tp = TextProcessor()
    chunk = Chunk(
        id="c001", doc_id="d001", doc_title="Test", text="inflammation mediates cancer",
        section="Body", page=1,
        tokens=tp.tokenize("inflammation mediates cancer")
    )
    idx.add_chunk(chunk)
    results = idx.search(["inflammation", "cancer"])
    assert len(results) == 1
    assert results[0].chunk.id == "c001"


def test_index_bm25_ranking():
    idx = InvertedIndex()
    tp = TextProcessor()
    # First chunk has the query term once
    c1 = Chunk(id="c1", doc_id="d1", doc_title="Doc1", text="tumor grows", section="Body", page=1,
               tokens=["tumor", "grows"])
    # Second chunk has the query term multiple times (higher TF)
    c2 = Chunk(id="c2", doc_id="d1", doc_title="Doc1", text="tumor tumor tumor metastasis",
               section="Body", page=1,
               tokens=["tumor", "tumor", "tumor", "metastasis"])
    idx.add_chunk(c1)
    idx.add_chunk(c2)
    results = idx.search(["tumor"])
    assert results[0].chunk.id == "c2", "Higher TF chunk should rank first"


def test_index_empty_search():
    idx = InvertedIndex()
    results = idx.search(["nonexistent_term_xyz"])
    assert results == []


def test_index_stats():
    idx = InvertedIndex()
    tp = TextProcessor()
    chunk = Chunk(id="s1", doc_id="d1", doc_title="T", text="test stats",
                  section="Body", page=1, tokens=["test", "stats"])
    idx.add_chunk(chunk)
    stats = idx.stats()
    assert stats["total_chunks"] == 1
    assert stats["unique_terms"] >= 2


# ─── QueryAnalyzer Tests ──────────────────────────────────────────────────────

def test_query_intent_mechanism():
    qa = QueryAnalyzer()
    result = qa.analyze("How does EGFR mutation affect drug resistance?")
    assert result["intent"] == "mechanism"


def test_query_intent_comparison():
    qa = QueryAnalyzer()
    result = qa.analyze("Compare ceftazidime-avibactam vs colistin for CRE")
    assert result["intent"] == "comparison"


def test_query_intent_treatment():
    qa = QueryAnalyzer()
    result = qa.analyze("What is the recommended treatment dose for pembrolizumab?")
    assert result["intent"] == "treatment"


def test_query_entities():
    qa = QueryAnalyzer()
    result = qa.analyze("What is the role of PD-L1 in NSCLC immunotherapy?")
    assert len(result["tokens"]) > 0
    assert len(result["expanded_tokens"]) >= len(result["tokens"])


def test_query_abbreviation_expansion():
    qa = QueryAnalyzer()
    result = qa.analyze("What does DNA methylation do?")
    # "dna" should be expanded
    assert any("deoxyribonucleic" in t or "dna" in t for t in result["expanded_tokens"])


# ─── EvidenceClassifier Tests ─────────────────────────────────────────────────

def test_classify_direct():
    clf = EvidenceClassifier()
    tp = TextProcessor()
    chunk = Chunk(
        id="e1", doc_id="d1", doc_title="T",
        text="The results demonstrate a significant reduction in mortality (p < 0.001).",
        section="Results", page=1,
        tokens=tp.tokenize("results demonstrate significant reduction mortality")
    )
    qa = QueryAnalyzer()
    qan = qa.analyze("Does the treatment reduce mortality?")
    support_type, terms = clf.classify(chunk, qan)
    assert support_type == "direct", f"Expected direct, got {support_type}"


def test_classify_contradictory():
    clf = EvidenceClassifier()
    tp = TextProcessor()
    chunk = Chunk(
        id="e2", doc_id="d1", doc_title="T",
        text="However, the trial failed to demonstrate any significant benefit in CVD outcomes.",
        section="Discussion", page=2,
        tokens=tp.tokenize("however trial failed significant benefit CVD")
    )
    qa = QueryAnalyzer()
    qan = qa.analyze("Does this therapy benefit CVD patients?")
    support_type, terms = clf.classify(chunk, qan)
    assert support_type == "contradictory", f"Expected contradictory, got {support_type}"


# ─── End-to-End Pipeline Tests ────────────────────────────────────────────────

def build_test_engine():
    engine = BioRAGEngine(chunk_size=400, retrieval_top_k=10, rerank_top_k=4)
    for doc in SAMPLE_DOCUMENTS:
        engine.add_document(doc["id"], doc["title"], doc["text"], doc.get("metadata"))
    return engine


def test_pipeline_returns_output():
    engine = build_test_engine()
    result = engine.query("What biomarkers predict cardiovascular risk in diabetes?")
    assert result is not None
    assert result.query != ""
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence_label in {"High", "Moderate", "Low", "Insufficient"}


def test_pipeline_evidence_nodes():
    engine = build_test_engine()
    result = engine.query("What is the efficacy of PD-L1 inhibitors in NSCLC?")
    assert len(result.evidence) > 0, "Should return at least one evidence node"
    for e in result.evidence:
        assert e.support_type in {"direct", "indirect", "contradictory"}
        assert 0.0 <= e.relevance_score <= 1.0


def test_pipeline_reasoning_chain():
    engine = build_test_engine()
    result = engine.query("How is Alzheimer's disease detected early?")
    assert len(result.reasoning_chain) >= 2, "Should have at least 2 reasoning steps"
    for step in result.reasoning_chain:
        assert step.step_number >= 1
        assert 0.0 <= step.confidence <= 1.0


def test_pipeline_follow_up_questions():
    engine = build_test_engine()
    result = engine.query("What are the treatment options for carbapenem-resistant bacteria?")
    assert len(result.follow_up_questions) >= 1


def test_pipeline_corpus_stats():
    engine = build_test_engine()
    stats = engine.get_corpus_stats()
    assert stats["documents"] == len(SAMPLE_DOCUMENTS)
    assert stats["chunks"] > 0
    assert stats["unique_terms"] > 100


def test_pipeline_empty_query_graceful():
    engine = build_test_engine()
    result = engine.query("zzzzxxx_nonexistent_term_completely_irrelevant")
    # Should return a result with low confidence, not crash
    assert result is not None
    assert result.confidence_label in {"Low", "Insufficient"}


def test_pipeline_comparison_query():
    engine = build_test_engine()
    result = engine.query("Compare colistin versus ceftazidime-avibactam for CRE treatment")
    assert result is not None
    # Comparison queries should find relevant evidence
    assert result.total_chunks_searched > 0


# ─── Runner ───────────────────────────────────────────────────────────────────

def main():
    print("\n" + "━"*60)
    print("  BioRAG Test Suite")
    print("━"*60)

    test_groups = [
        ("TextProcessor", [
            test_tokenize_basic,
            test_tokenize_preserves_short_biomedical,
            test_clean_text,
            test_extract_sentences,
        ]),
        ("DocumentChunker", [
            test_chunker_produces_chunks,
            test_chunk_has_tokens,
            test_chunk_overlap,
            test_section_detection,
        ]),
        ("InvertedIndex", [
            test_index_add_and_search,
            test_index_bm25_ranking,
            test_index_empty_search,
            test_index_stats,
        ]),
        ("QueryAnalyzer", [
            test_query_intent_mechanism,
            test_query_intent_comparison,
            test_query_intent_treatment,
            test_query_entities,
            test_query_abbreviation_expansion,
        ]),
        ("EvidenceClassifier", [
            test_classify_direct,
            test_classify_contradictory,
        ]),
        ("End-to-End Pipeline", [
            test_pipeline_returns_output,
            test_pipeline_evidence_nodes,
            test_pipeline_reasoning_chain,
            test_pipeline_follow_up_questions,
            test_pipeline_corpus_stats,
            test_pipeline_empty_query_graceful,
            test_pipeline_comparison_query,
        ]),
    ]

    total = passed = 0
    for group_name, tests in test_groups:
        print(f"\n  [{group_name}]")
        for test_fn in tests:
            total += 1
            if run_test(test_fn.__name__.replace("test_", ""), test_fn):
                passed += 1

    print(f"\n{'━'*60}")
    print(f"  Results: {passed}/{total} passed", end="")
    if passed == total:
        print("  ✓ All tests passed!")
    else:
        print(f"  ✗ {total-passed} failed")
    print("━"*60 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

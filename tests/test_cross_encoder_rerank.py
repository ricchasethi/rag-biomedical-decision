"""
Cross-Encoder Reranker Test Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for cross_encoder_rerank.py: CrossEncoderReranker scoring/ordering, the
sigmoid mapping and pair cache, plus end-to-end integration with BioRAGEngine
(lexical pre-filter → cross-encoder final ranking).

All tests use a FakeCrossEncoder (deterministic, offline) injected as the
reranker's loaded model, so the suite never downloads the real transformer.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_engine import BioRAGEngine, Chunk, RetrievedChunk
from cross_encoder_rerank import CrossEncoderReranker


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


# ─── Test Doubles & Helpers ────────────────────────────────────────────────────

class FakeCrossEncoder:
    """Deterministic, offline stand-in for sentence-transformers CrossEncoder.

    Scores each (query, text) pair by the count of shared lowercase tokens — a
    monotonic proxy for relevance. Tracks how many pairs were actually scored so
    tests can assert the pair cache prevents recomputation.
    """

    def __init__(self):
        self.pairs_scored = 0

    def predict(self, pairs, show_progress_bar: bool = False):
        self.pairs_scored += len(pairs)
        scores = []
        for q, t in pairs:
            shared = set(q.lower().split()) & set(t.lower().split())
            scores.append(float(len(shared)))
        return scores


def fake_reranker() -> CrossEncoderReranker:
    """A CrossEncoderReranker whose model is pre-loaded with the offline fake.

    Setting ``_model`` means ``_load()`` returns it without constructing the real
    transformer, keeping the test offline.
    """
    r = CrossEncoderReranker()
    r._model = FakeCrossEncoder()
    return r


def mk_retrieved(cid: str, text: str, rank: int, doc_id: str = "doc1") -> RetrievedChunk:
    chunk = Chunk(id=cid, doc_id=doc_id, doc_title="T", text=text, section="Results", page=1)
    return RetrievedChunk(chunk=chunk, score=1.0, rank=rank, match_terms=[cid])


# ─── 1. Sigmoid mapping ─────────────────────────────────────────────────────────

def test_sigmoid_midpoint():
    assert abs(CrossEncoderReranker._sigmoid(0.0) - 0.5) < 1e-9


def test_sigmoid_bounds_and_monotonic():
    # Large-magnitude logits stay strictly inside (0, 1) and don't overflow.
    lo = CrossEncoderReranker._sigmoid(-20.0)
    hi = CrossEncoderReranker._sigmoid(20.0)
    assert 0.0 < lo < 0.5 < hi < 1.0, f"sigmoid out of bounds: {lo}, {hi}"
    # No OverflowError at extreme inputs.
    assert CrossEncoderReranker._sigmoid(-1000.0) >= 0.0
    assert CrossEncoderReranker._sigmoid(1.0) > CrossEncoderReranker._sigmoid(-1.0)


# ─── 2. Reranking behaviour ─────────────────────────────────────────────────────

def test_rerank_orders_by_relevance():
    rr = fake_reranker()
    candidates = [
        mk_retrieved("c1", "lung cancer chemotherapy tumor", rank=1),
        mk_retrieved("c2", "plasma tau predicts alzheimer disease", rank=2),
    ]
    out = rr.rerank("plasma tau alzheimer", candidates, top_k=2)
    assert out[0].chunk.id == "c2", f"most relevant chunk should rank first, got {out[0].chunk.id}"


def test_rerank_scores_in_unit_interval():
    rr = fake_reranker()
    candidates = [
        mk_retrieved("c1", "alpha beta gamma", rank=1),
        mk_retrieved("c2", "delta epsilon", rank=2),
    ]
    out = rr.rerank("alpha beta", candidates, top_k=2)
    for r in out:
        assert 0.0 < r.score < 1.0, f"score {r.score} not in (0,1)"


def test_rerank_respects_top_k():
    rr = fake_reranker()
    candidates = [mk_retrieved(f"c{i}", f"text {i}", rank=i) for i in range(1, 6)]
    out = rr.rerank("text", candidates, top_k=3)
    assert len(out) == 3, f"expected 3 results, got {len(out)}"


def test_rerank_empty_returns_empty():
    rr = fake_reranker()
    assert rr.rerank("anything", [], top_k=5) == []


def test_rerank_assigns_sequential_ranks():
    rr = fake_reranker()
    candidates = [mk_retrieved(f"c{i}", f"shared text {i}", rank=i) for i in range(1, 5)]
    out = rr.rerank("shared text", candidates, top_k=4)
    assert [r.rank for r in out] == [1, 2, 3, 4], f"ranks not sequential: {[r.rank for r in out]}"


def test_rerank_preserves_match_terms():
    rr = fake_reranker()
    candidates = [mk_retrieved("c1", "plasma tau alzheimer", rank=1)]
    out = rr.rerank("plasma tau", candidates, top_k=1)
    assert out[0].match_terms == ["c1"], "match_terms from the candidate must be preserved"


def test_rerank_caches_repeated_pairs():
    rr = fake_reranker()
    candidates = [
        mk_retrieved("c1", "alpha beta gamma", rank=1),
        mk_retrieved("c2", "delta epsilon zeta", rank=2),
    ]
    rr.rerank("alpha beta", candidates, top_k=2)
    scored_after_first = rr._model.pairs_scored
    # Identical query + texts → served from cache, transformer not re-invoked.
    rr.rerank("alpha beta", candidates, top_k=2)
    assert rr._model.pairs_scored == scored_after_first, (
        f"cached pairs were re-scored: {rr._model.pairs_scored} > {scored_after_first}"
    )


# ─── 3. End-to-end with BioRAGEngine ────────────────────────────────────────────

ALZ_TEXT = (
    "Plasma p-tau217 is a biomarker that predicts Alzheimer's disease progression. "
    "Elevated p-tau217 correlates with amyloid burden in cerebrospinal fluid."
)
CANCER_TEXT = (
    "PD-L1 expression in lung cancer tumors correlates with immunotherapy response. "
    "Checkpoint inhibitors improve survival in non-small-cell lung carcinoma."
)


def test_engine_with_cross_encoder_returns_output():
    engine = BioRAGEngine(cross_encoder=fake_reranker())
    engine.add_document("d1", "Alz", ALZ_TEXT)
    out = engine.query("What plasma biomarkers predict Alzheimer's?")
    assert out.answer and out.confidence_label, "engine must return a valid DecisionOutput"
    assert isinstance(out.evidence, list)


def test_engine_cross_encoder_bounded_by_rerank_top_k():
    engine = BioRAGEngine(cross_encoder=fake_reranker(), rerank_top_k=2)
    engine.add_document("d1", "Alz", ALZ_TEXT)
    engine.add_document("d2", "Cancer", CANCER_TEXT)
    out = engine.query("plasma tau alzheimer disease biomarker")
    assert len(out.evidence) <= 2, f"evidence must respect rerank_top_k, got {len(out.evidence)}"
    # The cross-encoder ran (pairs scored) but never more than the pre-filter budget.
    scored = engine.cross_encoder._model.pairs_scored
    assert 0 < scored <= engine.cross_encoder_candidates, (
        f"cross-encoder scored {scored} pairs, expected 1..{engine.cross_encoder_candidates}"
    )


def test_engine_cross_encoder_picks_relevant_doc():
    engine = BioRAGEngine(cross_encoder=fake_reranker())
    engine.add_document("d1", "Alz", ALZ_TEXT)
    engine.add_document("d2", "Cancer", CANCER_TEXT)
    out = engine.query("plasma tau amyloid alzheimer progression")
    assert out.evidence, "expected at least one evidence node"
    assert out.evidence[0].doc_title == "Alz", (
        f"cross-encoder should surface the Alzheimer's doc first, got {out.evidence[0].doc_title}"
    )


def test_engine_without_cross_encoder_unchanged():
    engine = BioRAGEngine()
    assert engine.cross_encoder is None
    engine.add_document("d1", "Alz", ALZ_TEXT)
    out = engine.query("What plasma biomarkers predict Alzheimer's?")
    assert out.answer and out.confidence_label


# ─── Runner ─────────────────────────────────────────────────────────────────────

def main() -> int:
    test_groups = [
        ("Sigmoid mapping", [
            test_sigmoid_midpoint,
            test_sigmoid_bounds_and_monotonic,
        ]),
        ("Reranking behaviour", [
            test_rerank_orders_by_relevance,
            test_rerank_scores_in_unit_interval,
            test_rerank_respects_top_k,
            test_rerank_empty_returns_empty,
            test_rerank_assigns_sequential_ranks,
            test_rerank_preserves_match_terms,
            test_rerank_caches_repeated_pairs,
        ]),
        ("End-to-End Pipeline", [
            test_engine_with_cross_encoder_returns_output,
            test_engine_cross_encoder_bounded_by_rerank_top_k,
            test_engine_cross_encoder_picks_relevant_doc,
            test_engine_without_cross_encoder_unchanged,
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

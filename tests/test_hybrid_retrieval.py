"""
Hybrid Retrieval Test Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for hybrid_retrieval.py: EmbeddingModel, DenseRetriever, and
reciprocal_rank_fusion, plus end-to-end integration with BioRAGEngine.

Most tests use a FakeEmbeddingModel (deterministic, dependency-free) so the
suite stays fast and offline. The two EmbeddingModel tests load the real
sentence-transformers model once to verify dimension and caching behaviour.
"""

import sys
import os
import math
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_engine import BioRAGEngine, Chunk, RetrievedChunk
from hybrid_retrieval import EmbeddingModel, DenseRetriever, reciprocal_rank_fusion


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

class FakeEmbeddingModel:
    """Deterministic, dependency-free stand-in for EmbeddingModel.

    Embeds text as an L2-normalised token-hash bag-of-words vector so cosine
    similarity is meaningful (texts sharing tokens score higher). Tracks the
    total number of texts actually embedded so tests can assert that the
    DenseRetriever dedup guard prevents re-embedding already-stored chunks.
    """

    DIM = 64

    def __init__(self):
        self.texts_encoded = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.texts_encoded += len(texts)
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def mk_chunk(cid: str, text: str, doc_id: str = "doc1", section: str = "Results") -> Chunk:
    return Chunk(id=cid, doc_id=doc_id, doc_title="T", text=text, section=section, page=1)


def memory_retriever(model: FakeEmbeddingModel | None = None) -> DenseRetriever:
    """A DenseRetriever backed by an in-memory Qdrant collection."""
    return DenseRetriever(model or FakeEmbeddingModel(), qdrant_path=":memory:")


# Real model loaded at most once across the whole suite (it is ~400 MB).
_REAL_MODEL: EmbeddingModel | None = None


def real_model() -> EmbeddingModel:
    global _REAL_MODEL
    if _REAL_MODEL is None:
        _REAL_MODEL = EmbeddingModel()
    return _REAL_MODEL


# ─── 1. EmbeddingModel (real model) ─────────────────────────────────────────────

def test_encode_returns_correct_dimension():
    em = real_model()
    vecs = em.encode(["plasma biomarkers predict disease"])
    assert len(vecs) == 1, f"expected 1 vector, got {len(vecs)}"
    assert len(vecs[0]) == 768, f"expected 768-dim, got {len(vecs[0])}"


def test_encode_caches_repeated_text():
    em = real_model()
    em.encode(["a unique cache probe sentence"])
    # Replace the underlying transformer so any real re-encode raises loudly.
    loaded = em._load()
    original = loaded.encode

    def boom(*args, **kwargs):
        raise AssertionError("cached text was re-encoded through the transformer")

    loaded.encode = boom
    try:
        # Same text → served from cache, transformer not touched.
        cached = em.encode(["a unique cache probe sentence"])
        assert len(cached[0]) == 768
    finally:
        loaded.encode = original


# ─── 2. DenseRetriever (in-memory Qdrant) ───────────────────────────────────────

def test_add_chunks_upserts():
    dr = memory_retriever()
    dr.add_chunks([mk_chunk("c1", "alpha beta"), mk_chunk("c2", "gamma delta")])
    count = dr.client.count(DenseRetriever.COLLECTION).count
    assert count == 2, f"expected 2 points, got {count}"


def test_add_chunks_empty_is_noop():
    model = FakeEmbeddingModel()
    dr = memory_retriever(model)
    before = model.texts_encoded
    dr.add_chunks([])
    assert model.texts_encoded == before, "empty add_chunks should not embed anything"


def test_add_chunks_idempotent_skips_embedding():
    model = FakeEmbeddingModel()
    dr = memory_retriever(model)
    chunks = [mk_chunk("c1", "alpha beta"), mk_chunk("c2", "gamma delta")]
    dr.add_chunks(chunks)
    after_first = model.texts_encoded
    dr.add_chunks(chunks)  # identical chunks — should embed nothing new
    assert model.texts_encoded == after_first, (
        f"re-adding embedded {model.texts_encoded - after_first} extra texts"
    )
    assert dr.client.count(DenseRetriever.COLLECTION).count == 2


def test_add_chunks_mixed_only_embeds_new():
    model = FakeEmbeddingModel()
    dr = memory_retriever(model)
    dr.add_chunks([mk_chunk("c1", "alpha"), mk_chunk("c2", "beta")])
    before = model.texts_encoded
    dr.add_chunks([mk_chunk("c1", "alpha"), mk_chunk("c2", "beta"), mk_chunk("c3", "gamma")])
    assert model.texts_encoded - before == 1, (
        f"expected exactly 1 new embedding, got {model.texts_encoded - before}"
    )
    assert dr.client.count(DenseRetriever.COLLECTION).count == 3


def test_search_returns_sorted_tuples():
    dr = memory_retriever()
    dr.add_chunks([
        mk_chunk("c1", "alzheimer plasma biomarker tau"),
        mk_chunk("c2", "lung cancer tumor chemotherapy"),
    ])
    hits = dr.search("alzheimer plasma biomarker", top_k=2)
    assert len(hits) == 2
    for chunk_id, score in hits:
        assert isinstance(chunk_id, str) and isinstance(score, float)
    assert hits[0][1] >= hits[1][1], "hits must be sorted by score descending"
    assert hits[0][0] == "c1", f"expected alzheimer chunk first, got {hits[0][0]}"


def test_search_empty_collection_returns_empty():
    dr = memory_retriever()
    assert dr.search("anything") == []


# ─── 3. reciprocal_rank_fusion ──────────────────────────────────────────────────

def _bm25(ids: list[str], chunks: dict[str, Chunk]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=chunks[cid], score=float(len(ids) - i), rank=i + 1, match_terms=[cid])
        for i, cid in enumerate(ids)
    ]


def test_rrf_in_both_lists_ranks_higher():
    chunks = {c: mk_chunk(c, "x") for c in ("c1", "c2", "c3")}
    bm25 = _bm25(["c1", "c2"], chunks)
    dense = [("c2", 0.9), ("c3", 0.8)]
    fused = reciprocal_rank_fusion(bm25, dense, chunks, top_k=10)
    assert fused[0].chunk.id == "c2", f"c2 (in both) should rank first, got {fused[0].chunk.id}"


def test_rrf_k_changes_magnitude_not_order():
    chunks = {c: mk_chunk(c, "x") for c in ("c1", "c2", "c3")}
    bm25 = _bm25(["c1", "c2"], chunks)
    dense = [("c2", 0.9), ("c3", 0.8)]
    low = reciprocal_rank_fusion(bm25, dense, chunks, top_k=10, rrf_k=1)
    high = reciprocal_rank_fusion(bm25, dense, chunks, top_k=10, rrf_k=1000)
    assert [r.chunk.id for r in low] == [r.chunk.id for r in high], "ordering must be stable"
    assert low[0].score != high[0].score, "rrf_k must change score magnitude"


def test_rrf_capped_at_top_k():
    chunks = {c: mk_chunk(c, "x") for c in ("c1", "c2", "c3", "c4")}
    bm25 = _bm25(["c1", "c2", "c3", "c4"], chunks)
    fused = reciprocal_rank_fusion(bm25, [], chunks, top_k=2)
    assert len(fused) == 2, f"expected top_k=2 results, got {len(fused)}"


def test_rrf_preserves_match_terms_and_ranks():
    chunks = {c: mk_chunk(c, "x") for c in ("c1", "c2")}
    bm25 = _bm25(["c1", "c2"], chunks)
    fused = reciprocal_rank_fusion(bm25, [("c1", 0.5)], chunks, top_k=10)
    assert fused[0].rank == 1 and fused[-1].rank == len(fused), "ranks must be 1..n"
    top = next(r for r in fused if r.chunk.id == "c1")
    assert top.match_terms == ["c1"], "BM25 match_terms must be preserved"


def test_rrf_dense_only_missing_chunk_skipped():
    chunks = {c: mk_chunk(c, "x") for c in ("c1", "c2")}
    bm25 = _bm25(["c1", "c2"], chunks)
    # "c9" is in dense hits but absent from all_chunks → must be skipped, not crash.
    fused = reciprocal_rank_fusion(bm25, [("c9", 0.99)], chunks, top_k=10)
    assert all(r.chunk.id != "c9" for r in fused), "dense-only missing chunk must be skipped"


# ─── 4. End-to-end with BioRAGEngine ────────────────────────────────────────────

DOC_TEXT = (
    "Plasma p-tau217 is a biomarker that predicts Alzheimer's disease progression. "
    "Elevated p-tau217 correlates with amyloid burden in cerebrospinal fluid."
)


def test_engine_without_dense_unchanged():
    engine = BioRAGEngine()
    assert engine.dense_retriever is None
    engine.add_document("d1", "Alz", DOC_TEXT)
    out = engine.query("What plasma biomarkers predict Alzheimer's?")
    assert out.answer and out.confidence_label, "engine must return a valid DecisionOutput"


def test_engine_with_dense_returns_output():
    engine = BioRAGEngine(dense_retriever=memory_retriever())
    engine.add_document("d1", "Alz", DOC_TEXT)
    out = engine.query("What plasma biomarkers predict Alzheimer's?")
    assert out.answer and out.confidence_label
    assert isinstance(out.evidence, list)


def test_add_document_feeds_both_indexes():
    dr = memory_retriever()
    engine = BioRAGEngine(dense_retriever=dr)
    n = engine.add_document("d1", "Alz", DOC_TEXT)
    assert n > 0, "expected at least one chunk"
    assert engine.index.doc_count == 1, "BM25 index should hold the document"
    dense_count = dr.client.count(DenseRetriever.COLLECTION).count
    assert dense_count == n, f"dense index should hold all {n} chunks, got {dense_count}"


# ─── Runner ─────────────────────────────────────────────────────────────────────

def main() -> int:
    test_groups = [
        ("EmbeddingModel (real model)", [
            test_encode_returns_correct_dimension,
            test_encode_caches_repeated_text,
        ]),
        ("DenseRetriever", [
            test_add_chunks_upserts,
            test_add_chunks_empty_is_noop,
            test_add_chunks_idempotent_skips_embedding,
            test_add_chunks_mixed_only_embeds_new,
            test_search_returns_sorted_tuples,
            test_search_empty_collection_returns_empty,
        ]),
        ("reciprocal_rank_fusion", [
            test_rrf_in_both_lists_ranks_higher,
            test_rrf_k_changes_magnitude_not_order,
            test_rrf_capped_at_top_k,
            test_rrf_preserves_match_terms_and_ranks,
            test_rrf_dense_only_missing_chunk_skipped,
        ]),
        ("End-to-End Pipeline", [
            test_engine_without_dense_unchanged,
            test_engine_with_dense_returns_output,
            test_add_document_feeds_both_indexes,
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

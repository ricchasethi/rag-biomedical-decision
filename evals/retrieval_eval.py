"""
Retrieval Eval Harness — MRR and NDCG
═══════════════════════════════════════
Evaluates two pipeline stages independently:

  Stage A — BM25 retrieval  (InvertedIndex.search)
  Stage B — Reranker        (Reranker.rerank applied on top of Stage A)

Comparing them reveals whether the reranker actually improves ranking quality
over raw BM25 scores, and by how much.

Unit of evaluation: DOCUMENT (not chunk).
The index retrieves chunks; we aggregate per document by taking the maximum
BM25 score across all chunks that belong to the same document. This is called
max-pooling and is standard practice when the ground truth is document-level.

Metrics
───────
MRR@K  (Mean Reciprocal Rank at K)
    For each query, find the rank of the first relevant document in the top-K
    results. The reciprocal rank is 1/rank, or 0 if no relevant document
    appears in the top-K. MRR is the mean over all queries.
    Intuition: measures how quickly the system surfaces the FIRST useful result.
    Good for decision-support where users stop at the first confident answer.

NDCG@K  (Normalized Discounted Cumulative Gain at K)
    Accounts for the FULL ranked list and supports graded relevance (0/1/2).
    DCG@K = Σ (2^rel_i − 1) / log2(i+1)  for i = 1 … K
    IDCG@K = DCG of the ideal (perfect) ordering of the same relevance grades
    NDCG@K = DCG@K / IDCG@K  → always in [0, 1]
    Intuition: a highly relevant document at rank 1 scores more than at rank 3;
    partially relevant documents (grade 1) contribute less than grade-2 ones.
    Essential here because our ground truth distinguishes direct (2) from
    partial (1) relevance.

K values evaluated: 1, 3, 5
    K=1 — did the top result hit?
    K=3 — quality of the first screen of results
    K=5 — captures all 4 documents in the sample corpus, so acts as a ceiling
"""

import sys
import os
import math
import argparse
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_engine import BioRAGEngine, RetrievedChunk
from data.sample_corpus import SAMPLE_DOCUMENTS
from evals.ground_truth import EVAL_QUERIES, ALZHEIMER_QUERIES, RetrievalQuery


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class QueryMetrics:
    """Per-query metric snapshot for one retrieval stage."""
    rr: float                    # reciprocal rank (for MRR)
    ndcg: dict[int, float]       # k -> NDCG@k
    doc_ranking: list[tuple[str, float]]  # ordered (doc_id, score)


@dataclass
class QueryResult:
    """Full evaluation result for a single query."""
    query_id: str
    query: str
    intent: str
    relevant_docs: dict[str, int]
    bm25: QueryMetrics
    reranked: QueryMetrics


@dataclass
class IntentSummary:
    """Aggregate metrics for one intent category."""
    intent: str
    n_queries: int
    mrr_bm25: float
    mrr_reranked: float
    ndcg_bm25: dict[int, float]
    ndcg_reranked: dict[int, float]


@dataclass
class EvalReport:
    """Top-level report returned by RetrievalEvaluator.evaluate()."""
    n_queries: int
    ks: list[int]
    mrr_bm25: float
    mrr_reranked: float
    ndcg_bm25: dict[int, float]      # k -> mean NDCG@k across all queries
    ndcg_reranked: dict[int, float]
    per_query: list[QueryResult]
    by_intent: list[IntentSummary]


# The four retrieval modes compared in --hybrid mode, in display order.
HYBRID_MODES: list[tuple[str, str]] = [
    ("bm25", "BM25"),
    ("dense", "Dense"),
    ("hybrid", "Hybrid"),
    ("hybrid_rerank", "Hybrid+Rerank"),
]


@dataclass
class HybridQueryResult:
    """Per-query metrics for all four retrieval modes (only used with --hybrid)."""
    query_id: str
    query: str
    intent: str
    relevant_docs: dict[str, int]
    bm25: QueryMetrics            # BM25 only
    dense: QueryMetrics           # Qdrant cosine ANN only
    hybrid: QueryMetrics          # BM25 + dense fused via RRF, pre-rerank
    hybrid_rerank: QueryMetrics   # RRF fusion followed by the reranker


@dataclass
class HybridEvalReport:
    """Top-level report returned by RetrievalEvaluator.evaluate_hybrid()."""
    n_queries: int
    ks: list[int]
    mrr: dict[str, float]              # mode -> mean MRR@k_max across queries
    ndcg: dict[str, dict[int, float]]  # mode -> {k -> mean NDCG@k}
    per_query: list[HybridQueryResult]


# ─── Metric Functions ────────────────────────────────────────────────────────
#
# Each function is pure: it takes a ranked doc list + the relevance dict and
# returns a scalar. Keeping them separate from the evaluator class makes them
# easy to unit-test independently.

def reciprocal_rank(
    doc_ranking: list[tuple[str, float]],
    relevant_docs: dict[str, int],
    k: int,
) -> float:
    """Return 1/rank of the first relevant document in the top-K list.

    A document is relevant if its grade is > 0 in relevant_docs.
    Returns 0.0 if no relevant document appears within the top K positions.

    Why MRR instead of just Precision@K:
      In a decision-support context the clinician reads until they find a
      useful answer. MRR penalises a system that buries the answer at rank 3
      twice as much as one that surfaces it at rank 2.
    """
    for rank, (doc_id, _score) in enumerate(doc_ranking[:k], start=1):
        if relevant_docs.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def dcg_at_k(
    doc_ranking: list[tuple[str, float]],
    relevant_docs: dict[str, int],
    k: int,
) -> float:
    """Discounted Cumulative Gain at K.

    Formula: Σ (2^rel_i − 1) / log2(i+1)  for i in 1..K

    The exponential gain formula (2^rel − 1) means:
      grade 2 → gain of 3
      grade 1 → gain of 1
      grade 0 → gain of 0
    This amplifies the difference between direct (2) and partial (1) evidence,
    which matters for biomedical decision support where grade-2 docs are much
    more actionable than grade-1 docs.
    """
    gain = 0.0
    for rank, (doc_id, _score) in enumerate(doc_ranking[:k], start=1):
        rel = relevant_docs.get(doc_id, 0)
        if rel > 0:
            gain += (2 ** rel - 1) / math.log2(rank + 1)
    return gain


def idcg_at_k(relevant_docs: dict[str, int], k: int) -> float:
    """Ideal DCG at K — the DCG of the perfect ranking.

    Sorts all relevance grades descending and computes DCG as if the system
    had ranked them perfectly. This is the normalisation denominator for NDCG.

    A query with only one grade-2 document has IDCG@1 = 3.0 and
    IDCG@3 = 3.0 (only one relevant doc to place). NDCG@K then rewards any
    system that puts that document at rank 1.
    """
    sorted_grades = sorted(relevant_docs.values(), reverse=True)
    gain = 0.0
    for rank, rel in enumerate(sorted_grades[:k], start=1):
        if rel > 0:
            gain += (2 ** rel - 1) / math.log2(rank + 1)
    return gain


def ndcg_at_k(
    doc_ranking: list[tuple[str, float]],
    relevant_docs: dict[str, int],
    k: int,
) -> float:
    """Normalised DCG at K.

    Returns DCG@K / IDCG@K, or 0.0 if IDCG@K == 0 (no relevant documents
    exist for this query, which means the query should not be in the eval set).
    """
    ideal = idcg_at_k(relevant_docs, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(doc_ranking, relevant_docs, k) / ideal


# ─── Core Evaluator ──────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Evaluates BioRAG retrieval quality using MRR and NDCG.

    Uses the engine's existing components directly (index, query_analyzer,
    reranker) rather than the full engine.query() path, so we can measure
    each stage in isolation without the evidence classifier or synthesizer
    interfering with the result.

    Chunk-to-document aggregation (max-pooling)
    ───────────────────────────────────────────
    BM25 retrieves individual chunks; ground truth labels are at the document
    level. We resolve this by grouping chunks by their doc_id and keeping the
    highest-scored chunk per document. The document's rank is determined by
    this maximum score. Max-pooling is preferred over mean-pooling here because
    a single highly-matched chunk is a strong signal that the document is
    relevant, regardless of how many low-scoring chunks it also has.
    """

    def __init__(self, engine: BioRAGEngine, ks: list[int] | None = None):
        self.engine = engine
        self.ks = ks or [1, 3, 5]
        # Retrieve enough chunks to cover the full corpus at BM25 stage.
        # With ~4 documents and ~10 chunks each, 60 is a safe ceiling that
        # ensures every document has a chance to appear in the BM25 result set.
        self._bm25_top_k = 60

    # ── Chunk aggregation ────────────────────────────────────────────────────

    def _chunks_to_doc_ranking(
        self, chunks: list[RetrievedChunk]
    ) -> list[tuple[str, float]]:
        """Max-pool chunk scores to document level and return sorted list.

        Returns [(doc_id, max_score), ...] ordered highest score first.
        Documents not present in chunks are absent from the list (treated as
        rank ∞ when computing metrics).
        """
        doc_scores: dict[str, float] = {}
        for rc in chunks:
            doc_id = rc.chunk.doc_id
            doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), rc.score)
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # ── Per-query eval ───────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        doc_ranking: list[tuple[str, float]],
        relevant_docs: dict[str, int],
    ) -> QueryMetrics:
        """Compute RR and NDCG@K for a single stage result."""
        k_max = max(self.ks)
        return QueryMetrics(
            rr=reciprocal_rank(doc_ranking, relevant_docs, k=k_max),
            ndcg={k: ndcg_at_k(doc_ranking, relevant_docs, k) for k in self.ks},
            doc_ranking=doc_ranking,
        )

    def evaluate_query(self, rq: RetrievalQuery) -> QueryResult:
        """Run one query through BM25 then reranker, compute metrics for each.

        Pipeline:
          1. QueryAnalyzer expands the query tokens (abbreviation expansion,
             intent detection) — same pre-processing the engine uses at runtime.
          2. InvertedIndex.search returns top-N chunks by BM25 score.
          3. Chunk scores are max-pooled to document level → BM25 doc ranking.
          4. Reranker.rerank adjusts scores using section weights + term density.
          5. Reranked chunks are max-pooled → reranked doc ranking.
          6. MRR and NDCG are computed from each doc ranking against ground truth.
        """
        q_analysis = self.engine.query_analyzer.analyze(rq.query)

        # Stage A: BM25
        bm25_chunks = self.engine.index.search(
            q_analysis["expanded_tokens"],
            top_k=self._bm25_top_k,
        )
        bm25_doc_ranking = self._chunks_to_doc_ranking(bm25_chunks)

        # Stage B: Reranker (operates on the BM25 chunk list)
        reranked_chunks = self.engine.reranker.rerank(
            bm25_chunks,
            q_analysis,
            top_k=self.engine.rerank_top_k,
        )
        reranked_doc_ranking = self._chunks_to_doc_ranking(reranked_chunks)

        return QueryResult(
            query_id=rq.query_id,
            query=rq.query,
            intent=rq.intent,
            relevant_docs=rq.relevant_docs,
            bm25=self._compute_metrics(bm25_doc_ranking, rq.relevant_docs),
            reranked=self._compute_metrics(reranked_doc_ranking, rq.relevant_docs),
        )

    # ── Hybrid (four-mode) eval ──────────────────────────────────────────────

    def _dense_hits_to_doc_ranking(
        self, dense_hits: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Max-pool dense (chunk_id, cosine) hits to a document ranking.

        Mirrors _chunks_to_doc_ranking but for the raw (chunk_id, score) tuples
        returned by DenseRetriever.search. chunk_ids absent from the index (e.g.
        stale vectors) are skipped.
        """
        doc_scores: dict[str, float] = {}
        for chunk_id, score in dense_hits:
            chunk = self.engine.index.chunks.get(chunk_id)
            if chunk is None:
                continue
            doc_scores[chunk.doc_id] = max(doc_scores.get(chunk.doc_id, 0.0), score)
        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    def evaluate_query_hybrid(self, rq: RetrievalQuery) -> HybridQueryResult:
        """Run one query through all four modes and compute metrics for each.

        Requires self.engine.dense_retriever to be set. The four modes share the
        same BM25 candidate set and dense hits, so the comparison is apples-to-apples:
          1. BM25 only           — InvertedIndex.search, max-pooled.
          2. Dense only          — DenseRetriever.search, max-pooled.
          3. Hybrid (RRF)        — reciprocal_rank_fusion of (1) and (2), max-pooled.
          4. Hybrid + Rerank     — Reranker.rerank applied to the fused list.
        """
        from hybrid_retrieval import reciprocal_rank_fusion

        q_analysis = self.engine.query_analyzer.analyze(rq.query)

        # 1. BM25
        bm25_chunks = self.engine.index.search(
            q_analysis["expanded_tokens"], top_k=self._bm25_top_k
        )
        bm25_ranking = self._chunks_to_doc_ranking(bm25_chunks)

        # 2. Dense
        dense_hits = self.engine.dense_retriever.search(
            rq.query, top_k=self._bm25_top_k
        )
        dense_ranking = self._dense_hits_to_doc_ranking(dense_hits)

        # 3. Hybrid — RRF fusion (pre-rerank)
        fused = reciprocal_rank_fusion(
            bm25_chunks, dense_hits, self.engine.index.chunks, self._bm25_top_k
        )
        hybrid_ranking = self._chunks_to_doc_ranking(fused)

        # 4. Hybrid + Rerank
        reranked = self.engine.reranker.rerank(
            fused, q_analysis, top_k=self.engine.rerank_top_k
        )
        hybrid_rerank_ranking = self._chunks_to_doc_ranking(reranked)

        return HybridQueryResult(
            query_id=rq.query_id,
            query=rq.query,
            intent=rq.intent,
            relevant_docs=rq.relevant_docs,
            bm25=self._compute_metrics(bm25_ranking, rq.relevant_docs),
            dense=self._compute_metrics(dense_ranking, rq.relevant_docs),
            hybrid=self._compute_metrics(hybrid_ranking, rq.relevant_docs),
            hybrid_rerank=self._compute_metrics(hybrid_rerank_ranking, rq.relevant_docs),
        )

    def evaluate_hybrid(self, queries: list[RetrievalQuery]) -> HybridEvalReport:
        """Evaluate all queries across the four retrieval modes."""
        results = [self.evaluate_query_hybrid(q) for q in queries]
        n = len(results)
        mrr = {
            mode: sum(getattr(r, mode).rr for r in results) / n
            for mode, _ in HYBRID_MODES
        }
        ndcg = {
            mode: {
                k: sum(getattr(r, mode).ndcg[k] for r in results) / n
                for k in self.ks
            }
            for mode, _ in HYBRID_MODES
        }
        return HybridEvalReport(
            n_queries=n, ks=self.ks, mrr=mrr, ndcg=ndcg, per_query=results
        )

    # ── Aggregation ──────────────────────────────────────────────────────────

    def _aggregate(self, results: list[QueryResult]) -> tuple[float, float, dict, dict]:
        """Compute mean MRR and mean NDCG@K across a list of query results."""
        mrr_bm25 = sum(r.bm25.rr for r in results) / len(results)
        mrr_re   = sum(r.reranked.rr for r in results) / len(results)
        ndcg_bm25 = {
            k: sum(r.bm25.ndcg[k] for r in results) / len(results)
            for k in self.ks
        }
        ndcg_re = {
            k: sum(r.reranked.ndcg[k] for r in results) / len(results)
            for k in self.ks
        }
        return mrr_bm25, mrr_re, ndcg_bm25, ndcg_re

    def _by_intent(self, results: list[QueryResult]) -> list[IntentSummary]:
        """Break down metrics by QueryAnalyzer intent label."""
        groups: dict[str, list[QueryResult]] = {}
        for r in results:
            groups.setdefault(r.intent, []).append(r)

        summaries = []
        for intent, group in sorted(groups.items()):
            mrr_b, mrr_r, ndcg_b, ndcg_r = self._aggregate(group)
            summaries.append(IntentSummary(
                intent=intent,
                n_queries=len(group),
                mrr_bm25=mrr_b,
                mrr_reranked=mrr_r,
                ndcg_bm25=ndcg_b,
                ndcg_reranked=ndcg_r,
            ))
        return summaries

    # ── Main entry point ─────────────────────────────────────────────────────

    def evaluate(self, queries: list[RetrievalQuery]) -> EvalReport:
        """Evaluate all queries and return a structured report."""
        results = [self.evaluate_query(q) for q in queries]
        mrr_b, mrr_r, ndcg_b, ndcg_r = self._aggregate(results)
        return EvalReport(
            n_queries=len(results),
            ks=self.ks,
            mrr_bm25=mrr_b,
            mrr_reranked=mrr_r,
            ndcg_bm25=ndcg_b,
            ndcg_reranked=ndcg_r,
            per_query=results,
            by_intent=self._by_intent(results),
        )


# ─── Reporting ────────────────────────────────────────────────────────────────

def _delta(a: float, b: float) -> str:
    """Format the signed difference b − a."""
    d = b - a
    return f"{d:+.3f}"


def print_report(report: EvalReport, verbose: bool = False) -> None:
    """Print a human-readable eval report to stdout."""
    W = 68
    bar = "─" * W

    print(f"\n{'━' * W}")
    print(f"  BioRAG Retrieval Eval  ({report.n_queries} queries)")
    print(f"{'━' * W}")

    # ── Summary table ────────────────────────────────────────────────────────
    # Explains what each metric column means so output is self-documenting.
    print(f"\n  {'Metric':<16} {'BM25':>8} {'Reranked':>10} {'Δ':>8}")
    print(f"  {bar[:58]}")

    k_max = max(report.ks)
    print(f"  {'MRR@' + str(k_max):<16} {report.mrr_bm25:>8.3f} {report.mrr_reranked:>10.3f} {_delta(report.mrr_bm25, report.mrr_reranked):>8}")

    for k in report.ks:
        label = f"NDCG@{k}"
        b = report.ndcg_bm25[k]
        r = report.ndcg_reranked[k]
        print(f"  {label:<16} {b:>8.3f} {r:>10.3f} {_delta(b, r):>8}")

    print(f"\n  Note: Δ = Reranked − BM25  (positive = reranker improves ranking)")

    # ── By-intent breakdown ───────────────────────────────────────────────────
    # Stratifying by intent reveals which query types the pipeline handles well.
    # A reranker that helps for 'treatment' queries but hurts for 'mechanism'
    # queries needs intent-specific section weights (see Reranker.SECTION_WEIGHTS).
    print(f"\n  {'By Intent':<16} {'N':>3} {'BM25 MRR':>10} {'Reranked':>10} {'Δ':>8}")
    print(f"  {bar[:58]}")
    for s in report.by_intent:
        d = _delta(s.mrr_bm25, s.mrr_reranked)
        print(f"  {s.intent:<16} {s.n_queries:>3} {s.mrr_bm25:>10.3f} {s.mrr_reranked:>10.3f} {d:>8}")

    # ── Per-query detail ─────────────────────────────────────────────────────
    # Printed only with --verbose. Each row shows what the system returned and
    # how it compares to the ground truth, making it easy to diagnose failures.
    if verbose:
        print(f"\n  {'Per-Query Detail':}")
        print(f"  {bar}")
        for r in report.per_query:
            print(f"\n  [{r.query_id}] ({r.intent})  {r.query[:70]}")

            # Ground truth
            gt_str = ", ".join(
                f"{doc}(grade={g})" for doc, g in sorted(r.relevant_docs.items())
            )
            print(f"    Ground truth : {gt_str}")

            # BM25 top-3 docs
            bm25_top = ", ".join(
                f"{doc}({score:.2f})" for doc, score in r.bm25.doc_ranking[:3]
            ) or "—"
            print(f"    BM25 top-3   : {bm25_top}  →  RR={r.bm25.rr:.3f}  NDCG@3={r.bm25.ndcg.get(3, 0):.3f}")

            # Reranked top-3 docs
            re_top = ", ".join(
                f"{doc}({score:.2f})" for doc, score in r.reranked.doc_ranking[:3]
            ) or "—"
            print(f"    Reranked     : {re_top}  →  RR={r.reranked.rr:.3f}  NDCG@3={r.reranked.ndcg.get(3, 0):.3f}")

    print(f"\n{'━' * W}\n")


def print_hybrid_report(report: HybridEvalReport, verbose: bool = False) -> None:
    """Print the four-mode (BM25 / Dense / Hybrid / Hybrid+Rerank) comparison."""
    W = 78
    bar = "─" * W
    modes = HYBRID_MODES
    k_max = max(report.ks)

    print(f"\n{'━' * W}")
    print(f"  BioRAG Hybrid Retrieval Eval  ({report.n_queries} queries)")
    print(f"{'━' * W}")

    # ── Summary table: metric rows × four mode columns ────────────────────────
    header = f"  {'Metric':<10}" + "".join(f"{label:>16}" for _, label in modes)
    print(f"\n{header}")
    print(f"  {bar[:10 + 16 * len(modes)]}")

    mrr_row = f"  {'MRR@' + str(k_max):<10}" + "".join(
        f"{report.mrr[mode]:>16.3f}" for mode, _ in modes
    )
    print(mrr_row)

    for k in report.ks:
        row = f"  {'NDCG@' + str(k):<10}" + "".join(
            f"{report.ndcg[mode][k]:>16.3f}" for mode, _ in modes
        )
        print(row)

    # Deltas relative to the BM25 baseline make each stage's contribution visible.
    print(f"\n  Δ vs BM25 (MRR@{k_max}):", end="  ")
    base = report.mrr["bm25"]
    print("  ".join(
        f"{label} {report.mrr[mode] - base:+.3f}"
        for mode, label in modes if mode != "bm25"
    ))

    # ── Per-query MRR@k_max across the four modes ─────────────────────────────
    if verbose:
        print(f"\n  {'Per-Query MRR@' + str(k_max):}")
        print(f"  {bar}")
        qhdr = f"  {'Query':<22}" + "".join(f"{label:>14}" for _, label in modes)
        print(qhdr)
        print(f"  {bar}")
        for r in report.per_query:
            label = f"{r.query_id} ({r.intent})"[:21]
            row = f"  {label:<22}" + "".join(
                f"{getattr(r, mode).rr:>14.3f}" for mode, _ in modes
            )
            print(row)

    print(f"\n{'━' * W}\n")


# ─── Engine builder ───────────────────────────────────────────────────────────

def build_engine() -> BioRAGEngine:
    """Load the sample corpus into a fresh BioRAGEngine.

    retrieval_top_k is set high (60) so the evaluator's BM25 search can pull
    chunks from all documents; the engine's own rerank_top_k (5) is kept as
    the reranker budget, matching production settings.
    """
    engine = BioRAGEngine(retrieval_top_k=60, rerank_top_k=5)
    for doc in SAMPLE_DOCUMENTS:
        engine.add_document(
            doc["id"], doc["title"], doc["text"], doc.get("metadata", {})
        )
    return engine


def build_hybrid_engine() -> BioRAGEngine:
    """Load the sample corpus into a hybrid (BM25 + dense) BioRAGEngine.

    Uses an in-memory Qdrant collection so the eval always reflects the current
    corpus exactly (no stale vectors carried over between runs). Embedding the
    sample corpus on each run is cheap given its size.
    """
    from hybrid_retrieval import EmbeddingModel, DenseRetriever

    dense_retriever = DenseRetriever(EmbeddingModel(), qdrant_path=":memory:")
    engine = BioRAGEngine(
        retrieval_top_k=60, rerank_top_k=5, dense_retriever=dense_retriever
    )
    for doc in SAMPLE_DOCUMENTS:
        engine.add_document(
            doc["id"], doc["title"], doc["text"], doc.get("metadata", {})
        )
    return engine


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="BioRAG retrieval eval — MRR and NDCG"
    )
    parser.add_argument(
        "--alzheimer-only",
        action="store_true",
        help="Run only the Alzheimer's disease query subset (Q01–Q07)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-query detail: top-3 retrieved docs vs ground truth",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        metavar="K",
        help="K values for NDCG@K (default: 1 3 5)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Compare four modes: BM25 / Dense / Hybrid (RRF) / Hybrid+Rerank",
    )
    args = parser.parse_args()

    queries = ALZHEIMER_QUERIES if args.alzheimer_only else EVAL_QUERIES

    print("Loading corpus…", end=" ", flush=True)
    engine = build_hybrid_engine() if args.hybrid else build_engine()
    stats = engine.get_corpus_stats()
    print(f"done  ({stats['documents']} docs, {stats['chunks']} chunks, {stats['unique_terms']} terms)")

    evaluator = RetrievalEvaluator(engine, ks=args.ks)

    print(f"Evaluating {len(queries)} queries…")
    if args.hybrid:
        report = evaluator.evaluate_hybrid(queries)
        print_hybrid_report(report, verbose=args.verbose)
    else:
        report = evaluator.evaluate(queries)
        print_report(report, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())

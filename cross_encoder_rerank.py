"""
Cross-encoder reranking for BioRAG.

Adds a true semantic second-stage reranker that scores each ``(query, chunk)``
pair jointly with a cross-encoder transformer, replacing the lexical heuristic
in ``core.rag_engine.Reranker`` as the *final* ranking signal. The heuristic
reranker is kept upstream as a cheap pre-filter that cuts the candidate set
before the (more expensive) cross-encoder runs:

    BM25 → dense → RRF → Reranker (pre-filter, top-N) → CrossEncoderReranker (top-k)

Like ``DenseRetriever`` and ``ClaudeSynthesizer``, this component is **optional
and injected** via ``BioRAGEngine(cross_encoder=...)`` — ``core/rag_engine.py``
never imports this module except lazily inside ``query()``, so the core stays
stdlib-only.

Why a cross-encoder beats the lexical reranker:
    BM25 and the heuristic reranker only reward *term overlap* and section/density
    signals. A cross-encoder feeds the query and chunk text through a transformer
    *together*, so it models their semantic interaction (paraphrase, negation,
    entity disambiguation) and emits a single calibrated relevance score.
"""

from __future__ import annotations

import hashlib
import math

from sentence_transformers import CrossEncoder

from core.rag_engine import RetrievedChunk


# ─── Cross-Encoder Reranker ─────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Re-scores ``(query, chunk)`` pairs with a cross-encoder and returns the top-k.

    The model is lazy-loaded on the first ``rerank()`` call so importing this module
    (or constructing the object) stays cheap. Pair scores are cached by an MD5 of
    ``query + chunk_text`` so re-ranking the same pair (e.g. across eval runs in one
    process) never re-runs the transformer.

    Raw cross-encoder outputs are logits on an open scale; ``rerank`` maps them
    through a sigmoid to (0, 1) so the score is interpretable and the existing 0–1
    relevance normalisation in ``BioRAGEngine.query()`` works unchanged.
    """

    # MS-MARCO-tuned cross-encoder: small, fast, strong general reranking baseline.
    # Swap for a PubMedBERT cross-encoder (e.g. "ncbi/MedCPT-Cross-Encoder") for
    # tighter biomedical domain fit at the cost of a larger download.
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self._cache: dict[str, float] = {}

    def _load(self) -> CrossEncoder:
        """Load the underlying cross-encoder on first use."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Map a raw logit to (0, 1)."""
        # Guard against overflow for large-magnitude logits.
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        z = math.exp(x)
        return z / (1.0 + z)

    def _score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Return a sigmoid relevance score per text, serving cache hits directly.

        Only cache misses are passed to the transformer, in a single batched
        ``predict`` call, mirroring ``EmbeddingModel.encode``.
        """
        scores: list[float | None] = [None] * len(texts)
        to_score: list[str] = []
        indices: list[int] = []
        for i, text in enumerate(texts):
            key = hashlib.md5(f"{query}\x00{text}".encode()).hexdigest()
            if key in self._cache:
                scores[i] = self._cache[key]
            else:
                to_score.append(text)
                indices.append(i)

        if to_score:
            raw = self._load().predict(
                [(query, t) for t in to_score], show_progress_bar=False
            )
            for idx, logit, text in zip(indices, raw, to_score):
                s = self._sigmoid(float(logit))
                key = hashlib.md5(f"{query}\x00{text}".encode()).hexdigest()
                self._cache[key] = s
                scores[idx] = s

        return [s for s in scores]  # all positions now filled

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Re-score candidates with the cross-encoder and return the top-k.

        ``query`` is the raw user question (not the expanded BM25 token list) —
        the cross-encoder needs natural language to model query↔chunk interaction.
        The returned ``RetrievedChunk.score`` is the sigmoid relevance in (0, 1);
        ``match_terms`` is carried over from the input candidate unchanged.
        """
        if not candidates:
            return []

        pair_scores = self._score_pairs(query, [c.chunk.text for c in candidates])

        order = sorted(
            range(len(candidates)), key=lambda i: pair_scores[i], reverse=True
        )[:top_k]

        reranked: list[RetrievedChunk] = []
        for rank, i in enumerate(order, start=1):
            src = candidates[i]
            reranked.append(RetrievedChunk(
                chunk=src.chunk,
                score=pair_scores[i],
                rank=rank,
                match_terms=src.match_terms,
            ))
        return reranked

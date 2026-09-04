"""
Unified reranker module for BioRAG.

Collects the three final-stage reranking strategies behind one import point so
they can be compared and swapped freely. All three re-score the candidate set
that survives retrieval (BM25 → dense → RRF) and the cheap lexical pre-filter;
they differ only in *how* they score a (query, chunk) pair:

    ┌───────────────────┬──────────────────────────────────────────────────────┐
    │ Reranker          │ How it scores query↔chunk                             │
    ├───────────────────┼──────────────────────────────────────────────────────┤
    │ LexicalReranker   │ term overlap · section weight · discriminative-token  │
    │  (core, stdlib)   │ recall. No model. Bag-of-words — no semantics.        │
    │ BiEncoderReranker │ embed query and chunk SEPARATELY, rank by cosine.     │
    │                   │ Same model as hybrid dense retrieval; query never     │
    │                   │ sees the chunk during encoding.                       │
    │ CrossEncoderRerank│ embed query and chunk TOGETHER in one transformer     │
    │                   │ pass → one calibrated relevance score. Models          │
    │                   │ paraphrase / negation / entity disambiguation.        │
    └───────────────────┴──────────────────────────────────────────────────────┘

Interface
---------
`BiEncoderReranker` and `CrossEncoderReranker` share the injectable final-stage
signature the engine calls at `core.rag_engine.py`:

    rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]

so either is a drop-in for ``BioRAGEngine(cross_encoder=...)`` (the "final
reranker" slot). Both write a score in (0, 1) to ``RetrievedChunk.score`` and
carry ``match_terms`` over unchanged, so every downstream pipeline stage is
unaffected — exactly as documented for the cross-encoder.

``LexicalReranker`` is re-exported from ``core.rag_engine`` (it must live there:
it is the always-on pre-filter and the core stays stdlib-only). Note its
signature differs — ``rerank(results, query_analysis, top_k)`` — because it
consumes the analyzer's expanded tokens / intent rather than the raw question.
"""

from __future__ import annotations

import hashlib

# Re-export so all three rerankers are importable from one place.
from core.rag_engine import Reranker as LexicalReranker, RetrievedChunk
from cross_encoder_rerank import CrossEncoderReranker

__all__ = ["LexicalReranker", "BiEncoderReranker", "CrossEncoderReranker"]


# ─── Bi-Encoder Reranker ────────────────────────────────────────────────────────

class BiEncoderReranker:
    """
    Re-scores candidates by the cosine similarity between the query embedding and
    each chunk embedding, using the same bi-encoder as hybrid dense retrieval.

    Unlike ``DenseRetriever`` — which embeds every chunk once and does an ANN
    search over the *whole* corpus to *retrieve* — this reranker operates only on
    the already-retrieved candidate set, computing exact cosine (no ANN
    approximation) as the *final* ranking signal. Assessing it answers a distinct
    question: does the bi-encoder add value as a reranking stage, on top of the
    role it already plays inside RRF fusion?

    Reuses ``hybrid_retrieval.EmbeddingModel``. Pass the *same* instance the
    ``DenseRetriever`` uses to share its in-process vector cache; otherwise a
    private model is created and the ≤``cross_encoder_candidates`` candidate
    chunks are embedded on demand (cheap — a dozen short encodes per query).

    Query and chunk vectors are computed independently (that is what makes it a
    bi-encoder), so it cannot model query↔chunk interaction the way a
    cross-encoder can — but it is far cheaper and needs no second model download.
    """

    def __init__(self, embedding_model: "object | None" = None,
                 model_name: str | None = None):
        # Lazy import keeps this module importable without sentence-transformers
        # until a bi-encoder is actually constructed.
        from hybrid_retrieval import EmbeddingModel
        if embedding_model is not None:
            self.model = embedding_model
        elif model_name is not None:
            self.model = EmbeddingModel(model_name=model_name)
        else:
            self.model = EmbeddingModel()
        # Cache cosine scores by MD5(query + chunk_text), mirroring the
        # cross-encoder's pair cache so repeat pairs never re-embed.
        self._cache: dict[str, float] = {}

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity of two vectors, mapped from [-1, 1] to (0, 1)."""
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.5
        cos = dot / (na * nb)
        return (cos + 1.0) / 2.0  # -> (0, 1), consistent with the 0-1 relevance convention

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Re-score candidates by query↔chunk cosine and return the top-k.

        ``query`` is the raw user question (natural language), matching the
        cross-encoder's interface. The returned ``RetrievedChunk.score`` is the
        cosine mapped to (0, 1); ``match_terms`` is carried over unchanged.
        """
        if not candidates:
            return []

        # Resolve cache hits; embed the query + only the uncached chunk texts.
        texts = [c.chunk.text for c in candidates]
        scores: list[float | None] = [None] * len(texts)
        to_embed: list[str] = []
        to_embed_idx: list[int] = []
        for i, text in enumerate(texts):
            key = hashlib.md5(f"{query}\x00{text}".encode()).hexdigest()
            cached = self._cache.get(key)
            if cached is not None:
                scores[i] = cached
            else:
                to_embed.append(text)
                to_embed_idx.append(i)

        if to_embed:
            # One batched encode: query first, then the uncached chunk texts.
            vectors = self.model.encode([query] + to_embed)
            q_vec = vectors[0]
            for pos, i in enumerate(to_embed_idx):
                s = self._cosine(q_vec, vectors[pos + 1])
                key = hashlib.md5(f"{query}\x00{texts[i]}".encode()).hexdigest()
                self._cache[key] = s
                scores[i] = s

        order = sorted(
            range(len(candidates)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        reranked: list[RetrievedChunk] = []
        for rank, i in enumerate(order, start=1):
            src = candidates[i]
            reranked.append(RetrievedChunk(
                chunk=src.chunk,
                score=scores[i],
                rank=rank,
                match_terms=src.match_terms,
            ))
        return reranked

"""
Hybrid dense retrieval for BioRAG.

Adds semantic (embedding-based) retrieval alongside the stdlib BM25 index and
fuses the two result lists with Reciprocal Rank Fusion (RRF). These components
are injected into ``BioRAGEngine`` via the ``dense_retriever`` parameter, mirroring
the ``ClaudeSynthesizer`` injection pattern — ``core/rag_engine.py`` never imports
this module except lazily inside ``query()``.

Three components:
    EmbeddingModel          — wraps sentence-transformers, lazy-loads + caches.
    DenseRetriever          — owns a Qdrant collection of chunk vectors.
    reciprocal_rank_fusion  — merges BM25 + dense rankings into one list.
"""

from __future__ import annotations

import hashlib
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from core.rag_engine import Chunk, RetrievedChunk


# ─── Embedding Model ──────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Wraps a sentence-transformers model with lazy loading and an in-process cache.

    The model is only loaded on the first ``encode()`` call so importing this module
    (or constructing the object) stays cheap. Embeddings are cached by an MD5 of the
    input text, so repeated queries and re-indexing the same chunk never re-run the
    transformer.
    """

    DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._cache: dict[str, list[float]] = {}

    def _load(self) -> SentenceTransformer:
        """Load the underlying transformer on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts, returning one vector per input in the same order.

        Texts already present in the cache are served from it; only cache misses are
        passed to the transformer in a single batched call.
        """
        results: list[tuple[int, list[float]]] = []
        to_encode: list[str] = []
        indices: list[int] = []
        for i, t in enumerate(texts):
            key = hashlib.md5(t.encode()).hexdigest()
            if key in self._cache:
                results.append((i, self._cache[key]))
            else:
                to_encode.append(t)
                indices.append(i)
        if to_encode:
            vecs = self._load().encode(to_encode, show_progress_bar=False).tolist()
            for idx, vec, text in zip(indices, vecs, to_encode):
                key = hashlib.md5(text.encode()).hexdigest()
                self._cache[key] = vec
                results.append((idx, vec))
        results.sort(key=lambda x: x[0])
        return [v for _, v in results]


# ─── Dense Retriever ──────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Owns a Qdrant collection of chunk embeddings and answers cosine ANN queries.

    Persistence is file-based by default so embeddings survive process restarts
    (same rationale as ``--save-corpus`` for BM25). Pass ``qdrant_path=":memory:"``
    for an ephemeral in-memory collection (used by tests).
    """

    COLLECTION = "biorag_chunks"

    def __init__(
        self,
        model: EmbeddingModel,
        qdrant_path: str = "./qdrant_data",
    ):
        self.model = model
        self.client = QdrantClient(path=qdrant_path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist, sizing it to the model."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION not in existing:
            dim = len(self.model.encode(["probe"])[0])
            self.client.create_collection(
                self.COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def _point_id(self, chunk_id: str) -> str:
        """Deterministic UUID for a chunk id — lets us dedupe on re-index."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks, skipping any already stored in Qdrant.

        On every engine launch ``BioRAGEngine`` re-adds the sample corpus; this guard
        asks Qdrant which point IDs already exist (IDs only, no vectors fetched) and
        embeds only the genuinely new chunks, so the embedding model is hit once per
        chunk over the collection's lifetime.
        """
        if not chunks:
            return

        # Ask Qdrant which point IDs already exist — no vectors fetched, just IDs.
        candidate_ids = [self._point_id(c.id) for c in chunks]
        found = self.client.retrieve(
            collection_name=self.COLLECTION,
            ids=candidate_ids,
            with_vectors=False,
            with_payload=False,
        )
        already_indexed = {p.id for p in found}

        new_chunks = [
            c for c in chunks
            if self._point_id(c.id) not in already_indexed
        ]
        if not new_chunks:
            return  # all chunks already in Qdrant — nothing to embed

        vectors = self.model.encode([c.text for c in new_chunks])
        points = [
            PointStruct(
                id=self._point_id(c.id),
                vector=vec,
                payload={"chunk_id": c.id, "doc_id": c.doc_id, "section": c.section},
            )
            for c, vec in zip(new_chunks, vectors)
        ]
        self.client.upsert(collection_name=self.COLLECTION, points=points)

    def search(self, question: str, top_k: int = 15) -> list[tuple[str, float]]:
        """Return ``(chunk_id, cosine_score)`` tuples sorted by score descending."""
        vec = self.model.encode([question])[0]
        hits = self.client.query_points(
            collection_name=self.COLLECTION,
            query=vec,
            limit=top_k,
        ).points
        return [(h.payload["chunk_id"], h.score) for h in hits]


# ─── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: list[RetrievedChunk],
    dense_hits: list[tuple[str, float]],
    all_chunks: dict[str, Chunk],
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """
    Merge BM25 and dense results via Reciprocal Rank Fusion.

        rrf_score(doc) = Σ_i  1 / (k + rank_i)

    Documents absent from a list contribute 0 from that list. The fused score is
    written to ``RetrievedChunk.score``; the downstream 0–1 normalisation in
    ``BioRAGEngine.query()`` handles it transparently. ``match_terms`` is preserved
    from the BM25 result when available (dense-only hits get an empty list).
    """
    bm25_ranks = {r.chunk.id: (i + 1) for i, r in enumerate(bm25_results)}
    dense_ranks = {chunk_id: (i + 1) for i, (chunk_id, _) in enumerate(dense_hits)}

    all_ids = set(bm25_ranks) | set(dense_ranks)

    scores: dict[str, float] = {}
    for cid in all_ids:
        score = 0.0
        if cid in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[cid])
        if cid in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[cid])
        scores[cid] = score

    # Preserve existing RetrievedChunk objects to keep their match_terms.
    existing: dict[str, RetrievedChunk] = {r.chunk.id: r for r in bm25_results}

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

    fused: list[RetrievedChunk] = []
    for rank, cid in enumerate(sorted_ids, start=1):
        if cid in existing:
            rc = existing[cid]
            fused.append(RetrievedChunk(
                chunk=rc.chunk,
                score=scores[cid],
                rank=rank,
                match_terms=rc.match_terms,
            ))
        elif cid in all_chunks:
            fused.append(RetrievedChunk(
                chunk=all_chunks[cid],
                score=scores[cid],
                rank=rank,
                match_terms=[],
            ))
        # chunks only in the dense index but not in all_chunks are skipped

    return fused

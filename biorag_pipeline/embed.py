"""Embed stage: pending chunks -> vectors in Qdrant.

Fifth and final ingestion stage. Reads chunks with embedded=false, encodes them,
upserts into the Qdrant server, flags them embedded, and promotes any paper whose
chunks are now all present to status='indexed'.

    python -m biorag_pipeline.embed --limit 500
    python -m biorag_pipeline.embed --stats

Postgres stays the source of truth for what *should* exist; the chunks.embedded
column tracks what actually does. That split is what lets a half-finished
embedding run resume without re-encoding everything, and it is why papers are
promoted to 'indexed' by a database predicate rather than by this module
believing it succeeded.

The EmbeddingModel is reused from hybrid_retrieval.py rather than reimplemented -
same model, same MD5-keyed cache, so the CLI's --hybrid mode and this pipeline
can never drift onto different embeddings.
"""

from __future__ import annotations

import argparse
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from hybrid_retrieval import EmbeddingModel

from biorag_pipeline import db, repository as repo
from biorag_pipeline.config import CONFIG

ENCODE_BATCH = 64      # texts per transformer forward pass
UPSERT_BATCH = 256     # points per Qdrant request


@dataclass
class EmbedResult:
    n_pending: int = 0
    n_embedded: int = 0
    n_papers: int = 0
    n_replaced: int = 0    # papers whose stale vectors were dropped first
    n_promoted: int = 0    # papers advanced to 'indexed'
    n_failed: int = 0
    failed_ids: list[str] = field(default_factory=list)


class QdrantIndexer:
    """Owns the Qdrant collection that serves chunk vectors.

    Distinct from hybrid_retrieval.DenseRetriever, which targets an embedded
    file-based Qdrant for the single-process CLI. This one talks to the server
    over HTTP, carries a richer payload, and can delete a paper's points by
    filter - all things the pipeline needs and the CLI does not.
    """

    def __init__(self, url: str | None = None, collection: str | None = None,
                 model: EmbeddingModel | None = None):
        self.client = QdrantClient(url=url or CONFIG.qdrant_url)
        self.collection = collection or CONFIG.qdrant_collection
        self.model = model or EmbeddingModel(CONFIG.embed_model)

    def ensure_collection(self) -> int:
        """Create the collection if absent, sized to the model. Returns the dim.

        The vector size is probed from the model rather than hardcoded, so
        switching BIORAG_EMBED_MODEL to a 384-dim model needs no code change -
        though it does need a new collection, since dimensions are immutable.
        """
        dimension = len(self.model.encode(["probe"])[0])
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection not in existing:
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            # Without this index, delete-by-filter and any arxiv_id lookup degrade
            # to a full scan of the collection.
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="arxiv_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        return dimension

    @staticmethod
    def point_id(chunk_id: str) -> str:
        """Deterministic UUID for a chunk id.

        Qdrant point ids must be UUIDs or integers, and chunk ids are neither.
        uuid5 is a pure function of the input, so re-embedding the same chunk
        overwrites its point rather than adding a duplicate.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def delete_for_paper(self, arxiv_id: str) -> None:
        """Drop every point belonging to a paper.

        Called only when a paper's whole chunk set was just rewritten. Chunk ids
        are derived from character offsets, so new text produces different ids and
        the old points would otherwise linger as unreachable orphans.
        """
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[
                FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_id))
            ]),
        )

    def upsert(self, rows: list[dict[str, Any]]) -> int:
        """Encode and upsert chunk rows. Returns the number of points written.

        The chunk text is stored in the payload as well as in Postgres. It costs
        roughly 15% on top of a 768-dim float32 vector and makes the collection
        independently inspectable - you can read results in the Qdrant dashboard
        without joining back to the database.
        """
        written = 0
        for start in range(0, len(rows), UPSERT_BATCH):
            batch = rows[start:start + UPSERT_BATCH]
            vectors: list[list[float]] = []
            for chunk_start in range(0, len(batch), ENCODE_BATCH):
                sub = batch[chunk_start:chunk_start + ENCODE_BATCH]
                vectors.extend(self.model.encode([r["text"] for r in sub]))

            points = [
                PointStruct(
                    id=self.point_id(row["chunk_id"]),
                    vector=vector,
                    payload={
                        "chunk_id": row["chunk_id"],
                        "arxiv_id": row["arxiv_id"],
                        "doc_id": row["doc_id"],
                        "doc_title": row["doc_title"],
                        "section": row["section"],
                        "ordinal": row["ordinal"],
                        "page": row["page"],
                        "primary_category": row.get("primary_category"),
                        "published_at": (row["published_at"].isoformat()
                                         if row.get("published_at") else None),
                        "text": row["text"],
                    },
                )
                for row, vector in zip(batch, vectors)
            ]
            self.client.upsert(collection_name=self.collection, points=points)
            written += len(points)
        return written

    def count(self) -> int:
        """Number of points currently in the collection."""
        return self.client.count(self.collection, exact=True).count


def _group_by_paper(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["arxiv_id"]].append(row)
    return grouped


def embed_pending(run_id: str | None = None, limit: int | None = None,
                  indexer: QdrantIndexer | None = None) -> EmbedResult:
    """Embed every chunk with embedded=false, one paper at a time."""
    result = EmbedResult()
    indexer = indexer or QdrantIndexer()
    indexer.ensure_collection()

    with db.connect() as conn:
        pending = repo.pending_chunks(conn, limit=limit or 2000)
    result.n_pending = len(pending)
    if not pending:
        return result

    for arxiv_id, rows in _group_by_paper(pending).items():
        result.n_papers += 1
        try:
            with db.connect() as conn:
                total = repo.count_chunks(conn, arxiv_id)

            # Every chunk pending means the set was just (re)written. Any points
            # already in Qdrant for this paper are from the previous boundaries
            # and must go. A partial batch is a resumed run, where the existing
            # points are still valid and deleting them would lose work.
            if len(rows) == total:
                indexer.delete_for_paper(arxiv_id)
                result.n_replaced += 1

            written = indexer.upsert(rows)

            with db.connect() as conn:
                repo.mark_chunks_embedded(conn, [r["chunk_id"] for r in rows])
            result.n_embedded += written

        except Exception as exc:
            # Chunks stay embedded=false, so the next run retries this paper only.
            with db.connect() as conn:
                repo.record_error(conn, run_id, "embed", str(exc), arxiv_id)
            result.n_failed += 1
            result.failed_ids.append(arxiv_id)

    with db.connect() as conn:
        result.n_promoted = repo.promote_indexed(conn)

    return result


# --- CLI --------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(description="Embed pending chunks into Qdrant")
    ap.add_argument("--limit", type=int, default=2000,
                    help="maximum chunks to process in this run")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--stats", action="store_true",
                    help="report Postgres and Qdrant counts, embed nothing")
    ap.add_argument("--search", metavar="QUERY",
                    help="run a similarity search to sanity-check the index")
    args = ap.parse_args()

    indexer = QdrantIndexer()
    print(f"model      : {CONFIG.embed_model}")
    print(f"qdrant     : {CONFIG.qdrant_url}  collection={CONFIG.qdrant_collection}")

    if args.search:
        dimension = indexer.ensure_collection()
        vector = indexer.model.encode([args.search])[0]
        hits = indexer.client.query_points(
            collection_name=indexer.collection, query=vector, limit=5,
        ).points
        print(f"dim        : {dimension}\n\nresults for {args.search!r}:\n")
        for hit in hits:
            payload = hit.payload
            print(f"  {hit.score:.3f}  [{payload['section']}] "
                  f"{payload['doc_title'][:56]}")
            print(f"         {payload['text'][:150]}...\n")
        return 0

    if args.stats:
        with db.connect() as conn:
            stats = repo.corpus_stats(conn)
        print(f"\npostgres   : {stats['papers']} papers "
              f"({stats['papers_indexed']} indexed, {stats['papers_failed']} failed)")
        print(f"             {stats['chunks']} chunks "
              f"({stats['chunks_embedded']} embedded, "
              f"{stats['chunks_pending']} pending)")
        try:
            print(f"qdrant     : {indexer.count()} points")
        except Exception as exc:
            print(f"qdrant     : unavailable ({exc})")
        print()
        return 0

    dimension = indexer.ensure_collection()
    print(f"dim        : {dimension}\n")

    result = embed_pending(args.run_id, args.limit, indexer)

    print(f"pending    : {result.n_pending} chunks across {result.n_papers} paper(s)")
    print(f"  embedded : {result.n_embedded}")
    print(f"  replaced : {result.n_replaced} paper(s) had stale vectors dropped")
    print(f"  promoted : {result.n_promoted} paper(s) -> indexed")
    print(f"  failed   : {result.n_failed}")
    if result.failed_ids:
        print(f"             {', '.join(result.failed_ids[:5])}")
    print(f"\nqdrant     : {indexer.count()} points\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

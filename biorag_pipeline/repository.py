"""Persistence layer for the BioRAG ingestion pipeline.

Every database write in the pipeline goes through this module - DAG code never
writes ad-hoc SQL. Functions take an open connection so a caller can group several
operations into one transaction (see db.connect()).

This module deliberately knows nothing about core.rag_engine. Converting an engine
Chunk into a ChunkRecord is the indexer's job, which keeps persistence independently
testable and the retrieval engine swappable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import psycopg2.extras
from psycopg2.extensions import connection as PGConnection

UpsertResult = Literal["inserted", "updated", "unchanged"]
Stage = Literal["discover", "fetch", "parse", "chunk", "embed", "report", "cleanup"]
PaperStatus = Literal["discovered", "fetched", "parsed", "chunked",
                      "indexed", "failed", "skipped"]


# --- Records ----------------------------------------------------------------

@dataclass
class PaperRecord:
    """A paper as discovered from arXiv, before fetching or parsing."""

    arxiv_id: str
    title: str
    abstract: str | None = None
    version: int = 1
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    primary_category: str | None = None
    published_at: datetime | None = None
    updated_at_src: datetime | None = None
    pdf_url: str | None = None


@dataclass
class ChunkRecord:
    """One chunk of a paper, mirroring core.rag_engine.Chunk minus `tokens`."""

    chunk_id: str
    ordinal: int
    text: str
    section: str = "Body"
    page: int = 1
    char_start: int = 0
    char_end: int = 0
    n_tokens: int = 0


@dataclass
class RunCounters:
    """Per-run tallies, written back by finish_run() and read by the daily report."""

    n_discovered: int = 0
    n_fetched: int = 0
    n_parsed: int = 0
    n_indexed: int = 0
    n_chunks: int = 0
    n_embedded: int = 0
    n_failed: int = 0


# --- Papers -----------------------------------------------------------------

_UPSERT_PAPER = """
INSERT INTO papers (arxiv_id, version, title, abstract, authors, categories,
                    primary_category, published_at, updated_at_src, pdf_url,
                    first_run_id)
VALUES (%(arxiv_id)s, %(version)s, %(title)s, %(abstract)s, %(authors)s,
        %(categories)s, %(primary_category)s, %(published_at)s,
        %(updated_at_src)s, %(pdf_url)s, %(run_id)s)
ON CONFLICT (arxiv_id) DO UPDATE SET
    version          = EXCLUDED.version,
    title            = EXCLUDED.title,
    abstract         = EXCLUDED.abstract,
    authors          = EXCLUDED.authors,
    categories       = EXCLUDED.categories,
    primary_category = EXCLUDED.primary_category,
    published_at     = EXCLUDED.published_at,
    updated_at_src   = EXCLUDED.updated_at_src,
    pdf_url          = EXCLUDED.pdf_url,
    status           = 'discovered',
    error            = NULL
WHERE EXCLUDED.version > papers.version
RETURNING (xmax = 0) AS inserted
"""


def upsert_paper(
    conn: PGConnection, paper: PaperRecord, run_id: str | None = None
) -> UpsertResult:
    """Insert a paper, or refresh it only if arXiv published a newer version.

    Three outcomes, distinguished so the caller can count them for the daily report:

      "inserted"  - genuinely new paper
      "updated"   - a higher arXiv version arrived; status reset to 'discovered'
                    so the DAG re-fetches and re-chunks it
      "unchanged" - already known at this version or newer; nothing written

    The `xmax = 0` test is the standard Postgres idiom for telling an INSERT apart
    from an UPDATE inside a RETURNING clause: a freshly inserted row has no
    deleting-transaction id. The WHERE guard means an unchanged row is not
    rewritten at all, so re-running a DAG produces zero write amplification.
    """
    params: dict[str, Any] = {
        "arxiv_id": paper.arxiv_id,
        "version": paper.version,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors,
        "categories": paper.categories,
        "primary_category": paper.primary_category,
        "published_at": paper.published_at,
        "updated_at_src": paper.updated_at_src,
        "pdf_url": paper.pdf_url,
        "run_id": run_id,
    }
    with conn.cursor() as cur:
        cur.execute(_UPSERT_PAPER, params)
        row = cur.fetchone()
    if row is None:
        return "unchanged"
    return "inserted" if row[0] else "updated"


def mark_paper(
    conn: PGConnection,
    arxiv_id: str,
    status: PaperStatus,
    *,
    raw_path: str | None = None,
    content_hash: str | None = None,
    error: str | None = None,
) -> None:
    """Advance a paper's lifecycle status, optionally recording where it landed.

    Only non-None keyword arguments are written, so a later stage never clobbers a
    field an earlier one set. Setting status='indexed' also stamps indexed_at.
    """
    sets = ["status = %(status)s", "error = %(error)s"]
    params: dict[str, Any] = {"arxiv_id": arxiv_id, "status": status, "error": error}

    if raw_path is not None:
        sets.append("raw_path = %(raw_path)s")
        params["raw_path"] = raw_path
    if content_hash is not None:
        sets.append("content_hash = %(content_hash)s")
        params["content_hash"] = content_hash
    if status == "indexed":
        sets.append("indexed_at = now()")

    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE papers SET {', '.join(sets)} WHERE arxiv_id = %(arxiv_id)s",
            params,
        )


def get_paper(conn: PGConnection, arxiv_id: str) -> dict[str, Any] | None:
    """Return one paper row as a dict, or None if unknown."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM papers WHERE arxiv_id = %s", (arxiv_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def papers_by_status(
    conn: PGConnection, status: PaperStatus, limit: int = 500
) -> list[dict[str, Any]]:
    """Papers currently at a given lifecycle stage - how a failed run resumes."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM papers WHERE status = %s ORDER BY discovered_at LIMIT %s",
            (status, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def content_changed(conn: PGConnection, arxiv_id: str, content_hash: str) -> bool:
    """True when this text differs from what is already stored.

    Lets a re-fetched paper skip chunking and embedding entirely when arXiv's new
    version only changed metadata. Unknown papers count as changed.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT content_hash FROM papers WHERE arxiv_id = %s", (arxiv_id,))
        row = cur.fetchone()
    return row is None or row[0] != content_hash


# --- Chunks -----------------------------------------------------------------

_INSERT_CHUNKS = """
INSERT INTO chunks (chunk_id, arxiv_id, ordinal, text, section, page,
                    char_start, char_end, n_tokens)
VALUES %s
"""


def replace_chunks(
    conn: PGConnection, arxiv_id: str, chunks: list[ChunkRecord]
) -> int:
    """Atomically replace all chunks for a paper. Returns the number written.

    Delete-then-insert rather than upsert: when a paper's text changes, the new
    chunk boundaries may not line up with the old ones, so stale rows must go.
    Both statements run in the caller's transaction, so a crash mid-way leaves the
    previous chunk set intact.

    Note this orphans the corresponding vectors in Qdrant. The indexer (Step 5)
    deletes points by chunk_id before calling this; Step 8's cleanup sweeps any
    that slipped through.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE arxiv_id = %s", (arxiv_id,))
        if chunks:
            rows = [
                (c.chunk_id, arxiv_id, c.ordinal, c.text, c.section, c.page,
                 c.char_start, c.char_end, c.n_tokens)
                for c in chunks
            ]
            # execute_values batches into a single multi-row INSERT - roughly an
            # order of magnitude faster than executemany for hundreds of chunks.
            psycopg2.extras.execute_values(cur, _INSERT_CHUNKS, rows, page_size=500)
        cur.execute(
            "UPDATE papers SET n_chunks = %s WHERE arxiv_id = %s",
            (len(chunks), arxiv_id),
        )
    return len(chunks)


def pending_chunks(conn: PGConnection, limit: int = 1000) -> list[dict[str, Any]]:
    """Chunks written to Postgres but not yet confirmed in Qdrant.

    Served by the `chunks_pending_idx` partial index, so this stays fast no matter
    how large the corpus grows. Joins papers for the title and generated doc_id,
    which the embedder needs to build the Qdrant payload.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT c.chunk_id, c.arxiv_id, c.ordinal, c.text, c.section, c.page,
                   c.char_start, c.char_end, c.n_tokens,
                   p.doc_id, p.title AS doc_title, p.primary_category,
                   p.published_at
            FROM chunks c
            JOIN papers p USING (arxiv_id)
            WHERE NOT c.embedded
            ORDER BY c.arxiv_id, c.ordinal
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def mark_chunks_embedded(conn: PGConnection, chunk_ids: list[str]) -> int:
    """Flag chunks as present in Qdrant. Returns the number updated."""
    if not chunk_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE chunks SET embedded = TRUE WHERE chunk_id = ANY(%s)",
            (chunk_ids,),
        )
        return cur.rowcount


def count_chunks(conn: PGConnection, arxiv_id: str) -> int:
    """Total chunks stored for a paper, embedded or not.

    The embedder compares this against how many of the paper's chunks are pending.
    When they are equal the chunk set was just rewritten, so its stale Qdrant
    points must be dropped before the new ones go in.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE arxiv_id = %s", (arxiv_id,))
        return cur.fetchone()[0]


def promote_indexed(conn: PGConnection) -> int:
    """Advance every fully-embedded 'chunked' paper to 'indexed'.

    A paper is only indexed once *all* of its chunks are confirmed in Qdrant, so a
    partial embedding run leaves it at 'chunked' and the next run finishes the job.
    Returns the number of papers promoted.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE papers p
               SET status = 'indexed', indexed_at = now()
             WHERE p.status = 'chunked'
               AND EXISTS (SELECT 1 FROM chunks c WHERE c.arxiv_id = p.arxiv_id)
               AND NOT EXISTS (SELECT 1 FROM chunks c
                                WHERE c.arxiv_id = p.arxiv_id AND NOT c.embedded)
            """
        )
        return cur.rowcount


# --- Runs and errors --------------------------------------------------------

def start_run(
    conn: PGConnection,
    run_id: str,
    dag_id: str,
    logical_date: datetime,
    *,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    categories: list[str] | None = None,
) -> None:
    """Open (or reopen) a run row.

    ON CONFLICT resets started_at and status, so clearing and re-running a task in
    the Airflow UI does not trip the primary key.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingest_runs (run_id, dag_id, logical_date,
                                     window_start, window_end, categories)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                started_at  = now(),
                finished_at = NULL,
                status      = 'running'
            """,
            (run_id, dag_id, logical_date, window_start, window_end,
             categories or []),
        )


def finish_run(
    conn: PGConnection,
    run_id: str,
    status: Literal["success", "failed"],
    counters: RunCounters,
) -> None:
    """Close a run and record its tallies. Read by the Step 7 daily report."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ingest_runs SET
                finished_at  = now(),
                status       = %s,
                n_discovered = %s,
                n_fetched    = %s,
                n_parsed     = %s,
                n_indexed    = %s,
                n_chunks     = %s,
                n_embedded   = %s,
                n_failed     = %s
            WHERE run_id = %s
            """,
            (status, counters.n_discovered, counters.n_fetched, counters.n_parsed,
             counters.n_indexed, counters.n_chunks, counters.n_embedded,
             counters.n_failed, run_id),
        )


def record_error(
    conn: PGConnection,
    run_id: str | None,
    stage: Stage,
    reason: str,
    arxiv_id: str | None = None,
) -> None:
    """Record a per-paper failure as data rather than raising.

    One unparseable PDF must never fail a whole run. The daily report groups these
    by stage. Call start_run() first - run_id is a foreign key.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingest_errors (run_id, arxiv_id, stage, reason)
            VALUES (%s, %s, %s, %s)
            """,
            (run_id, arxiv_id, stage, reason[:4000]),
        )


def run_summary(conn: PGConnection, run_id: str) -> dict[str, Any]:
    """Counters for one run, derived entirely from the database.

    The DAG's final task uses this instead of collecting XComs from the upstream
    stages. XComs vanish when a task fails, which is exactly when the summary
    matters most; the database still holds the truth either way. It also keeps the
    same numbers available to anyone querying Postgres directly.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
              (SELECT count(*) FROM papers
                WHERE first_run_id = r.run_id)                     AS n_discovered,
              (SELECT count(*) FROM papers
                WHERE status = 'fetched' OR raw_path IS NOT NULL
                  AND discovered_at >= r.started_at)               AS n_fetched,
              (SELECT count(*) FROM papers
                WHERE status = 'indexed'
                  AND indexed_at >= r.started_at)                  AS n_indexed,
              (SELECT count(*) FROM chunks c JOIN papers p USING (arxiv_id)
                WHERE p.indexed_at >= r.started_at)                AS n_chunks,
              (SELECT count(*) FROM chunks c JOIN papers p USING (arxiv_id)
                WHERE c.embedded AND p.indexed_at >= r.started_at) AS n_embedded,
              (SELECT count(*) FROM ingest_errors
                WHERE run_id = r.run_id)                           AS n_failed
            FROM ingest_runs r
            WHERE r.run_id = %s
            """,
            (run_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else {}


def corpus_stats(conn: PGConnection) -> dict[str, Any]:
    """Corpus-wide counts, for the daily report and health checks."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
              (SELECT count(*) FROM papers)                        AS papers,
              (SELECT count(*) FROM papers WHERE status='indexed') AS papers_indexed,
              (SELECT count(*) FROM papers WHERE status='failed')  AS papers_failed,
              (SELECT count(*) FROM chunks)                        AS chunks,
              (SELECT count(*) FROM chunks WHERE embedded)         AS chunks_embedded,
              (SELECT count(*) FROM chunks WHERE NOT embedded)     AS chunks_pending
            """
        )
        return dict(cur.fetchone())

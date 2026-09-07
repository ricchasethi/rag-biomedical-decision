"""Chunk stage: parsed text -> rows in the chunks table.

Fourth stage of the DAG. Reads papers at status='parsed', splits their extracted
text with the engine's own DocumentChunker, and writes the result to Postgres.

    python -m biorag_pipeline.chunk --limit 10
    python -m biorag_pipeline.chunk --arxiv-id 2609.01055 --show

This module is the seam promised in Step 2: repository.py knows nothing about
core.rag_engine, and core.rag_engine knows nothing about Postgres. The conversion
from the engine's Chunk to the repository's ChunkRecord happens here and nowhere
else, so either side can be replaced without touching the other.

Chunking deliberately reuses DocumentChunker rather than reimplementing it. The
chunk ids it produces are md5(doc_id:char_offset)[:12] - deterministic, so
re-chunking identical text yields identical ids and the Qdrant upsert downstream
overwrites in place instead of duplicating.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from core.rag_engine import DocumentChunker
from biorag_pipeline import db, repository as repo
from biorag_pipeline.config import CONFIG
from biorag_pipeline.fetch import raw_paths
from biorag_pipeline.parse import sanitize_text
from biorag_pipeline.repository import ChunkRecord


@dataclass
class ChunkResult:
    n_attempted: int = 0
    n_chunked: int = 0
    n_chunks: int = 0
    n_empty: int = 0           # produced no chunks at all
    n_failed: int = 0
    n_rechunked: int = 0       # replaced an existing chunk set
    chunked_ids: list[str] = field(default_factory=list)


def build_chunker() -> DocumentChunker:
    """DocumentChunker configured from the environment.

    Kept separate so the DAG, the CLI and tests all get identical settings -
    chunk_size drives retrieval quality, and a silent mismatch between stages
    would be very hard to notice.
    """
    return DocumentChunker(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )


def chunk_text(doc_id: str, title: str, text: str,
               chunker: DocumentChunker | None = None) -> list[ChunkRecord]:
    """Split text into ChunkRecords using the engine's chunker.

    `tokens` is not carried across: it is a pure function of
    TextProcessor.tokenize() and would go stale the moment clean_text() changes.
    Only the count is kept, as a cheap sanity signal.
    """
    chunker = chunker or build_chunker()
    chunks = chunker.chunk_document(doc_id, title, text)
    return [
        ChunkRecord(
            chunk_id=c.id,
            ordinal=i,
            text=c.text,
            section=c.section,
            page=c.page,
            char_start=c.char_start,
            char_end=c.char_end,
            n_tokens=len(c.tokens),
        )
        for i, c in enumerate(chunks)
    ]


def chunk_pending(run_id: str | None = None, limit: int | None = None,
                  arxiv_ids: list[str] | None = None) -> ChunkResult:
    """Chunk every paper at status='parsed' and advance it to 'chunked'."""
    result = ChunkResult()
    chunker = build_chunker()

    with db.connect() as conn:
        if arxiv_ids:
            papers = [p for p in (repo.get_paper(conn, i) for i in arxiv_ids) if p]
        else:
            papers = repo.papers_by_status(conn, "parsed", limit=limit or 500)

    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        _, text_path = raw_paths(arxiv_id)
        result.n_attempted += 1

        if not text_path.exists():
            # The .txt was deleted (retention sweep, manual cleanup). Send the
            # paper back to 'fetched' so parse regenerates it rather than failing
            # it permanently - the PDF is usually still on disk.
            with db.connect() as conn:
                repo.mark_paper(conn, arxiv_id, "fetched")
                repo.record_error(conn, run_id, "chunk",
                                  f"missing text file {text_path}", arxiv_id)
            result.n_failed += 1
            continue

        try:
            # sanitize_text again on read: .txt files written before the parser
            # learned to strip control characters still contain them, and a single
            # NUL byte makes Postgres reject the whole batch insert.
            text = sanitize_text(text_path.read_text(encoding="utf-8"))
            records = chunk_text(paper["doc_id"], paper["title"], text, chunker)
        except Exception as exc:
            with db.connect() as conn:
                repo.mark_paper(conn, arxiv_id, "failed", error=f"chunking: {exc}")
                repo.record_error(conn, run_id, "chunk", str(exc), arxiv_id)
            result.n_failed += 1
            continue

        if not records:
            with db.connect() as conn:
                repo.mark_paper(conn, arxiv_id, "failed",
                                error="chunker produced no chunks")
                repo.record_error(conn, run_id, "chunk",
                                  "chunker produced no chunks", arxiv_id)
            result.n_empty += 1
            continue

        had_chunks = (paper.get("n_chunks") or 0) > 0

        # replace_chunks and mark_paper share one transaction: a paper is never
        # left at 'chunked' with a half-written chunk set.
        #
        # The write is guarded as well as the chunking. A value the database
        # refuses - a NUL byte surviving extraction, an over-long field - is bad
        # data from a PDF, not a programmer error, so it must fail this paper
        # rather than the whole batch of five hundred.
        try:
            with db.connect() as conn:
                repo.replace_chunks(conn, arxiv_id, records)
                repo.mark_paper(conn, arxiv_id, "chunked")
        except Exception as exc:
            with db.connect() as conn:
                repo.mark_paper(conn, arxiv_id, "failed", error=f"chunk write: {exc}")
                repo.record_error(conn, run_id, "chunk", str(exc), arxiv_id)
            result.n_failed += 1
            continue

        result.n_chunked += 1
        result.n_chunks += len(records)
        result.n_rechunked += int(had_chunks)
        result.chunked_ids.append(arxiv_id)

    return result


# --- CLI --------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(description="Chunk parsed papers into Postgres")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--arxiv-id", action="append",
                    help="chunk specific ids instead of the pending queue")
    ap.add_argument("--show", action="store_true",
                    help="print a section breakdown and the first chunks")
    args = ap.parse_args()

    print(f"chunk_size={CONFIG.chunk_size} overlap={CONFIG.chunk_overlap}")

    result = chunk_pending(args.run_id, args.limit, args.arxiv_id)

    print(f"attempted  : {result.n_attempted}")
    print(f"  chunked  : {result.n_chunked}  ({result.n_rechunked} re-chunked)")
    print(f"  chunks   : {result.n_chunks}")
    print(f"  empty    : {result.n_empty}")
    print(f"  failed   : {result.n_failed}")
    if result.n_chunked:
        print(f"  avg/paper: {result.n_chunks / result.n_chunked:.1f}")

    if args.show and result.chunked_ids:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT section, count(*), round(avg(length(text))) AS avg_chars
                  FROM chunks WHERE arxiv_id = ANY(%s)
              GROUP BY section ORDER BY 2 DESC
                """,
                (result.chunked_ids,),
            )
            print("\nsections across this batch:")
            for section, count, avg_chars in cur.fetchall():
                print(f"  {section:<16} {count:>4} chunks  avg {avg_chars} chars")

            cur.execute(
                """
                SELECT ordinal, section, left(text, 160)
                  FROM chunks WHERE arxiv_id = %s ORDER BY ordinal LIMIT 3
                """,
                (result.chunked_ids[0],),
            )
            print(f"\nfirst chunks of {result.chunked_ids[0]}:")
            for ordinal, section, preview in cur.fetchall():
                print(f"\n  [{ordinal}] ({section})\n  {preview}...")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

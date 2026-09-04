"""Discovery stage: search arXiv and record what was found in Postgres.

The first stage of the daily DAG. It writes to ingest_runs and papers only - no
PDFs are downloaded and no text is parsed - so it is cheap, fast, and safe to
re-run at any time.

    python -m biorag_pipeline.discover --days 3 --max-results 10
    python -m biorag_pipeline.discover --since 2026-09-01 --until 2026-09-04
    python -m biorag_pipeline.discover --days 3 --dry-run

Two properties worth knowing about:

* Network calls never happen inside a database transaction. arXiv paging with
  retries can take a minute; holding a Postgres transaction open that long blocks
  vacuum and risks an idle-in-transaction timeout. So: open a connection to record
  the run, close it, search, then open a second connection for the writes.

* One malformed paper cannot abort the batch. Each upsert runs inside its own
  SAVEPOINT, so a failure rolls back that paper alone and the remaining forty-nine
  still commit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone

from biorag_pipeline import db, repository as repo
from biorag_pipeline.arxiv_client import (
    SEARCH_FIELDS,
    ArxivError,
    default_window,
    fetch_by_ids,
    search,
)
from biorag_pipeline.config import CONFIG


@dataclass
class DiscoveryResult:
    """What one discovery pass found and wrote."""

    run_id: str
    window_start: datetime | None
    window_end: datetime | None
    categories: list[str]
    n_found: int = 0          # returned by arXiv
    n_inserted: int = 0       # genuinely new papers
    n_updated: int = 0        # a newer arXiv version arrived
    n_unchanged: int = 0      # already stored at this version - no write performed
    n_failed: int = 0         # rejected by the database, recorded in ingest_errors
    pending: list[str] = field(default_factory=list)

    @property
    def n_written(self) -> int:
        """Papers that actually need fetching downstream."""
        return self.n_inserted + self.n_updated


def discover(
    run_id: str,
    *,
    dag_id: str = "arxiv_ingest_daily",
    logical_date: datetime | None = None,
    categories: list[str] | None = None,
    lookback_days: int = 2,
    max_results: int | None = None,
    window: tuple[datetime, datetime] | None = None,
    terms: list[str] | None = None,
    field: str = "all",
    arxiv_ids: list[str] | None = None,
) -> DiscoveryResult:
    """Discover papers and upsert them into the papers table.

    Three selection modes, checked in this order:

      arxiv_ids  - exact papers by identifier; category, topic and window ignored
      terms      - topic search, optionally narrowed by categories and window
      categories - the daily mode: everything new in these categories

    Pass categories=[] together with terms to search all of arXiv for the topic
    rather than only the configured categories.

    Returns a DiscoveryResult whose `pending` list holds the papers that changed.
    Step 4 does not rely on that list to decide what to fetch - it queries
    papers_by_status(conn, 'discovered') instead, so a run that crashed after
    discovery resumes correctly rather than re-searching arXiv.

    Raises ArxivError if the API is unreachable; the failure is recorded in
    ingest_errors first so the daily report still sees it.
    """
    if categories is None and not terms and not arxiv_ids:
        categories = CONFIG.arxiv_categories
    categories = categories or []
    logical_date = logical_date or datetime.now(timezone.utc)

    if arxiv_ids:
        # An explicit id list is not a time window; leave the run's window null so
        # the daily report does not read it as coverage of those dates.
        window_start = window_end = None
    else:
        window_start, window_end = window or default_window(lookback_days, logical_date)

    result = DiscoveryResult(
        run_id=run_id,
        window_start=window_start,
        window_end=window_end,
        categories=categories,
    )

    # --- 1. Record the run, then close the connection before touching the network.
    with db.connect() as conn:
        repo.start_run(
            conn, run_id, dag_id, logical_date,
            window_start=window_start,
            window_end=window_end,
            categories=categories,
        )

    # --- 2. Search arXiv with no transaction held open.
    try:
        if arxiv_ids:
            papers = fetch_by_ids(arxiv_ids)
        else:
            papers = search(categories, window_start, window_end,
                            max_results=max_results, terms=terms, field=field)
    except ArxivError as exc:
        with db.connect() as conn:
            repo.record_error(conn, run_id, "discover", str(exc))
        raise

    result.n_found = len(papers)

    # --- 3. Write, one SAVEPOINT per paper.
    with db.connect() as conn:
        for paper in papers:
            with conn.cursor() as cur:
                cur.execute("SAVEPOINT paper_sp")
            try:
                outcome = repo.upsert_paper(conn, paper.to_record(), run_id)
                with conn.cursor() as cur:
                    cur.execute("RELEASE SAVEPOINT paper_sp")
            except Exception as exc:
                # Without the rollback the connection stays in a failed-transaction
                # state and every later statement raises too, losing the batch.
                with conn.cursor() as cur:
                    cur.execute("ROLLBACK TO SAVEPOINT paper_sp")
                    cur.execute("RELEASE SAVEPOINT paper_sp")
                repo.record_error(conn, run_id, "discover", str(exc), paper.arxiv_id)
                result.n_failed += 1
                continue

            if outcome == "inserted":
                result.n_inserted += 1
                result.pending.append(paper.arxiv_id)
            elif outcome == "updated":
                result.n_updated += 1
                result.pending.append(paper.arxiv_id)
            else:
                result.n_unchanged += 1

    return result


def close_run(result: DiscoveryResult, status: str = "success") -> None:
    """Mark a discovery-only run finished. The Step 6 DAG does this itself, at the
    end of the whole pipeline, with counters from every stage."""
    with db.connect() as conn:
        repo.finish_run(
            conn, result.run_id, status,          # type: ignore[arg-type]
            repo.RunCounters(
                n_discovered=result.n_found,
                n_failed=result.n_failed,
            ),
        )


# --- CLI --------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(description="Discover arXiv papers into Postgres")
    ap.add_argument("--categories", default=",".join(CONFIG.arxiv_categories))
    ap.add_argument("--days", type=int, default=2, help="lookback window (default: 2)")
    ap.add_argument("--since", help="window start YYYY-MM-DD (overrides --days)")
    ap.add_argument("--until", help="window end YYYY-MM-DD")
    ap.add_argument("--max-results", type=int, default=CONFIG.arxiv_max_results)
    ap.add_argument("--run-id", help="defaults to manual__<timestamp>")
    ap.add_argument("--search", action="append", metavar="PHRASE",
                    help="topic phrase; repeat to AND several together")
    ap.add_argument("--field", default="all", choices=sorted(SEARCH_FIELDS),
                    help="which field --search looks in (default: all)")
    ap.add_argument("--arxiv-id", action="append", metavar="ID",
                    help="ingest these exact ids; ignores topic, category and dates")
    ap.add_argument("--any-category", action="store_true",
                    help="search all of arXiv, not just the configured categories")
    ap.add_argument("--dry-run", action="store_true",
                    help="search and report, write nothing")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    run_id = args.run_id or f"manual__{now.strftime('%Y%m%dT%H%M%S')}"

    if args.since:
        start = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        end = (datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc)
               if args.until else now)
        window = (start, end)
    else:
        window = default_window(args.days, now)

    categories = ([] if args.any_category
                  else [c.strip() for c in args.categories.split(",") if c.strip()])

    print(f"run_id    : {run_id}")
    if args.arxiv_id:
        print(f"id_list   : {', '.join(args.arxiv_id)}")
    else:
        print(f"categories: {', '.join(categories) or '(all of arXiv)'}")
        if args.search:
            print(f"topic     : {args.field}:{' AND '.join(args.search)}")
        print(f"window    : {window[0].isoformat()}  ->  {window[1].isoformat()}")

    if args.dry_run:
        if args.arxiv_id:
            papers = fetch_by_ids(args.arxiv_id)
        else:
            papers = search(categories, window[0], window[1],
                            max_results=args.max_results,
                            terms=args.search, field=args.field)
        print(f"\ndry run - {len(papers)} paper(s) found, nothing written\n")
        for p in papers:
            print(f"  {p.arxiv_id:<18} v{p.version}  [{p.primary_category}]  "
                  f"{p.title[:70]}")
        return 0

    result = discover(
        run_id,
        categories=categories,
        max_results=args.max_results,
        window=window,
        terms=args.search,
        field=args.field,
        arxiv_ids=args.arxiv_id,
    )
    close_run(result)

    print(f"\nfound     : {result.n_found}")
    print(f"  inserted: {result.n_inserted}")
    print(f"  updated : {result.n_updated}")
    print(f"  unchanged: {result.n_unchanged}")
    print(f"  failed  : {result.n_failed}")
    print(f"\n{result.n_written} paper(s) need fetching\n")

    with db.connect() as conn:
        stats = repo.corpus_stats(conn)
    print(f"corpus    : {stats['papers']} papers, {stats['chunks']} chunks\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

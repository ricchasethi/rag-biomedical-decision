"""Daily arXiv ingestion DAG.

Deliberately thin. Every task is a wrapper that calls one function in
biorag_pipeline/ and returns a small dict; no pipeline logic lives in this file.
That is what lets the whole pipeline be developed and tested from a shell without
Airflow running, and it keeps DAG parsing fast.

    discover -> fetch -> parse (mapped) -> chunk -> embed -> finish
       ^
    migrate

Design notes:

* migrate runs first, every time. Re-running is a no-op, and it means a schema
  change can never be deployed after the code that depends on it.

* Heavy imports live inside task bodies, not at module level. The dag-processor
  re-imports this file every few seconds; importing torch here would make every
  parse cycle take seconds.

* No stage takes its work list from the previous stage's XCom. Each queries
  Postgres by status, so a run that dies mid-way resumes from the database rather
  than needing a list from a task that already failed.

* finish reads its counters from the database rather than from upstream XComs.
  XComs disappear when a task fails, which is precisely when the summary matters.
"""

from __future__ import annotations

import pendulum
from airflow.sdk import Param, dag, get_current_context, task

from biorag_pipeline.config import CONFIG

PARSE_BATCH_SIZE = 5

DEFAULT_ARGS = {
    "retries": 2,
    "retry_delay": pendulum.duration(minutes=5),
}


@dag(
    dag_id="arxiv_ingest_daily",
    description="Fetch, parse, chunk and embed new arXiv papers into BioRAG",
    schedule="0 6 * * *",                       # 06:00 UTC daily
    start_date=pendulum.datetime(2026, 9, 1, tz="UTC"),
    catchup=False,                              # never backfill on first deploy
    max_active_runs=1,                          # two runs would fight over queues
    default_args=DEFAULT_ARGS,
    tags=["biorag", "ingestion"],
    doc_md=__doc__,
    params={
        "categories": Param(
            ",".join(CONFIG.arxiv_categories), type="string",
            description="Comma-separated arXiv categories. Empty = all of arXiv."),
        "search": Param(
            "", type="string",
            description="Optional topic phrase, AND-ed with the categories."),
        "lookback_days": Param(
            2, type="integer", minimum=1, maximum=365,
            description="Submission window width. Wider than a day on purpose: "
                        "arXiv's index lags, and duplicates are free."),
        "max_results": Param(
            CONFIG.arxiv_max_results, type="integer", minimum=1, maximum=2000),
    },
)
def arxiv_ingest_daily():

    @task
    def migrate_schema() -> int:
        """Reconcile the database schema. No-op when already current."""
        from biorag_pipeline.migrate import migrate
        return migrate()

    @task
    def discover_papers() -> dict:
        """Search arXiv for the run's window and upsert into papers."""
        from biorag_pipeline.discover import discover

        context = get_current_context()
        params = context["params"]
        categories = [c.strip() for c in params["categories"].split(",") if c.strip()]
        terms = [params["search"]] if params["search"].strip() else None

        # Airflow 3 made logical_date optional for manual runs, and when it is
        # absent the key is missing from the context entirely rather than being
        # None - so this must be .get(), not a subscript. A run triggered from the
        # CLI without --logical-date has none; one triggered from the UI does.
        logical_date = context.get("logical_date") or pendulum.now("UTC")

        result = discover(
            run_id=context["run_id"],
            dag_id="arxiv_ingest_daily",
            logical_date=logical_date,
            categories=categories,
            lookback_days=params["lookback_days"],
            max_results=params["max_results"],
            terms=terms,
        )
        return {
            "found": result.n_found,
            "inserted": result.n_inserted,
            "updated": result.n_updated,
            "unchanged": result.n_unchanged,
        }

    @task
    def fetch_papers(_upstream: dict) -> dict:
        """Download PDFs for everything at status='discovered'.

        Deliberately NOT mapped. The arXiv rate limit is enforced by a
        process-local throttle, so parallel mapped instances would each start
        their own timer and hammer the API. One task, one throttle.
        """
        from biorag_pipeline.fetch import fetch_pending

        context = get_current_context()
        result = fetch_pending(run_id=context["run_id"], limit=500)
        return {
            "fetched": result.n_fetched,
            "cached": result.n_cached,
            "failed": result.n_failed,
        }

    @task
    def parse_batches(_upstream: dict) -> list[list[str]]:
        """Split the papers awaiting parsing into batches for dynamic mapping."""
        from biorag_pipeline import db, repository as repo

        with db.connect() as conn:
            papers = repo.papers_by_status(conn, "fetched", limit=500)
        ids = [p["arxiv_id"] for p in papers]
        return [ids[i:i + PARSE_BATCH_SIZE]
                for i in range(0, len(ids), PARSE_BATCH_SIZE)]

    @task(max_active_tis_per_dag=4)
    def parse_papers(arxiv_ids: list[str]) -> dict:
        """Extract text from one batch of PDFs.

        Mapped, unlike fetch: pypdf extraction is CPU-bound with no external rate
        limit, so parallelism is a real speedup. Batched rather than one task per
        paper because 50 task instances cost more in scheduler overhead than the
        extraction itself.
        """
        from biorag_pipeline.parse import parse_pending

        context = get_current_context()
        result = parse_pending(run_id=context["run_id"], arxiv_ids=arxiv_ids)
        return {
            "parsed": result.n_parsed,
            "unchanged": result.n_unchanged,
            "abstract_only": result.n_abstract_only,
            "failed": result.n_failed,
        }

    @task
    def chunk_papers(_upstream: list[dict]) -> dict:
        """Chunk parsed text into the chunks table. Fast and IO-bound; one task."""
        from biorag_pipeline.chunk import chunk_pending

        context = get_current_context()
        result = chunk_pending(run_id=context["run_id"], limit=500)
        return {
            "chunked": result.n_chunked,
            "chunks": result.n_chunks,
            "rechunked": result.n_rechunked,
            "failed": result.n_failed,
        }

    @task
    def embed_chunks(_upstream: dict) -> dict:
        """Embed pending chunks into Qdrant.

        Not mapped: the sentence-transformers model is ~420 MB and would be loaded
        once per mapped instance. One task loads it once and batches internally.
        """
        from biorag_pipeline.embed import embed_pending

        context = get_current_context()
        result = embed_pending(run_id=context["run_id"], limit=5000)
        return {
            "embedded": result.n_embedded,
            "promoted": result.n_promoted,
            "replaced": result.n_replaced,
            "failed": result.n_failed,
        }

    @task(trigger_rule="all_done")
    def finish(_upstream: dict) -> dict:
        """Close the run row with counters read from the database.

        trigger_rule='all_done' so the run is always closed, even when an upstream
        stage failed - an ingest_runs row stuck at 'running' would be worse than
        one marked 'failed'.
        """
        from biorag_pipeline import db, repository as repo

        context = get_current_context()
        run_id = context["run_id"]

        with db.connect() as conn:
            summary = repo.run_summary(conn, run_id)
            counters = repo.RunCounters(
                n_discovered=summary.get("n_discovered", 0),
                n_fetched=summary.get("n_fetched", 0),
                n_indexed=summary.get("n_indexed", 0),
                n_chunks=summary.get("n_chunks", 0),
                n_embedded=summary.get("n_embedded", 0),
                n_failed=summary.get("n_failed", 0),
            )
            status = "failed" if counters.n_failed and not counters.n_indexed \
                else "success"
            repo.finish_run(conn, run_id, status, counters)
            stats = repo.corpus_stats(conn)

        return {"run": summary, "corpus": stats}

    # --- Wiring -------------------------------------------------------------
    # Each stage takes the previous one's result only to express ordering; none
    # of them uses it as a work list.
    schema = migrate_schema()
    discovered = discover_papers()
    fetched = fetch_papers(discovered)
    parsed = parse_papers.expand(arxiv_ids=parse_batches(fetched))
    chunked = chunk_papers(parsed)
    embedded = embed_chunks(chunked)
    finish(embedded)

    schema >> discovered


arxiv_ingest_daily()

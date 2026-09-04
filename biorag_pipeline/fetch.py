"""Fetch stage: download PDFs for discovered papers into var/raw/.

Second stage of the DAG. Reads papers with status='discovered', downloads each
PDF, and advances them to 'fetched'.

    python -m biorag_pipeline.fetch --limit 5
    python -m biorag_pipeline.fetch --arxiv-id 2609.01055

Three things this stage is careful about:

* Transient vs permanent failure. A timeout leaves the paper at 'discovered' so
  tomorrow's run retries it; a 404 or a non-PDF response marks it 'failed' so it
  is not retried forever. Both are recorded in ingest_errors either way.

* Atomic writes. The download goes to a .part file and is renamed only after the
  magic bytes are verified, so a crash mid-download can never leave a truncated
  file that looks like a valid cached PDF.

* Politeness. arXiv asks for no more than one request every few seconds. For bulk
  historical downloads use their S3 access instead of this endpoint.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from biorag_pipeline import db, repository as repo
from biorag_pipeline.config import CONFIG

USER_AGENT = "BioRAG/0.1 (biomedical decision-support RAG; +https://arxiv.org/help/api)"

FETCH_DELAY = 3.0            # seconds between downloads
TIMEOUT = 60
MAX_PDF_BYTES = 50 * 1024 * 1024
PDF_MAGIC = b"%PDF-"


class FetchError(RuntimeError):
    """A download failed. `permanent` decides whether the paper is retried."""

    def __init__(self, message: str, *, permanent: bool = False):
        super().__init__(message)
        self.permanent = permanent


@dataclass
class FetchResult:
    n_attempted: int = 0
    n_fetched: int = 0
    n_cached: int = 0          # already on disk, download skipped
    n_failed: int = 0
    n_permanent: int = 0
    fetched_ids: list[str] = field(default_factory=list)


# --- Paths ------------------------------------------------------------------

def shard_for(arxiv_id: str) -> str:
    """Directory shard derived from the identifier itself.

    Modern ids encode YYMM: '2609.01234' -> '2026-09'. Pre-2007 ids ('math.GT/
    0309136') have no such prefix and go to 'legacy'. Deriving the shard from the
    id rather than from the ingest date keeps it stable across re-runs, so a
    re-fetch always resolves to the same path.
    """
    head = arxiv_id.split("/")[0] if "/" in arxiv_id else arxiv_id.split(".")[0]
    if len(head) == 4 and head.isdigit():
        return f"20{head[:2]}-{head[2:]}"
    return "legacy"


def safe_name(arxiv_id: str) -> str:
    """Filesystem-safe stem. Pre-2007 ids contain a slash."""
    return arxiv_id.replace("/", "_")


def raw_paths(arxiv_id: str) -> tuple[Path, Path]:
    """Return (pdf_path, text_path) for a paper. The text path is where the parse
    stage writes extracted text, so re-chunking never re-reads the PDF."""
    directory = CONFIG.raw_dir / shard_for(arxiv_id)
    stem = safe_name(arxiv_id)
    return directory / f"{stem}.pdf", directory / f"{stem}.txt"


# --- Throttling -------------------------------------------------------------

_last_request = 0.0
_throttle_lock = threading.Lock()


def _throttle(min_interval: float = FETCH_DELAY) -> None:
    global _last_request
    with _throttle_lock:
        wait = min_interval - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


# --- Download ---------------------------------------------------------------

def download_pdf(url: str, destination: Path,
                 session: requests.Session | None = None) -> int:
    """Stream a PDF to `destination`, returning its size in bytes.

    Writes to a .part file and renames on success. os.replace is atomic within a
    filesystem, so a reader can never observe a partial file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    owns_session = session is None
    session = session or requests.Session()
    if owns_session:
        session.headers.update({"User-Agent": USER_AGENT})

    try:
        _throttle()
        with session.get(url, stream=True, timeout=TIMEOUT) as response:
            if response.status_code in (403, 404, 410):
                raise FetchError(f"HTTP {response.status_code} for {url}",
                                 permanent=True)
            response.raise_for_status()

            written = 0
            with open(partial, "wb") as handle:
                for block in response.iter_content(chunk_size=65536):
                    written += len(block)
                    if written > MAX_PDF_BYTES:
                        raise FetchError(
                            f"pdf exceeds {MAX_PDF_BYTES // 1024 // 1024} MB",
                            permanent=True)
                    handle.write(block)

        # arXiv sometimes answers with an HTML "PDF is being generated" page and
        # a 200 status. Checking the magic bytes is the only reliable test.
        with open(partial, "rb") as handle:
            if handle.read(len(PDF_MAGIC)) != PDF_MAGIC:
                raise FetchError("response is not a PDF", permanent=False)

        os.replace(partial, destination)
        return written

    except requests.RequestException as exc:
        raise FetchError(f"download failed: {exc}", permanent=False) from exc
    finally:
        partial.unlink(missing_ok=True)
        if owns_session:
            session.close()


# --- Stage ------------------------------------------------------------------

def fetch_pending(run_id: str | None = None, limit: int | None = None,
                  arxiv_ids: list[str] | None = None) -> FetchResult:
    """Download every paper still at status='discovered'.

    Deliberately reads the work list from Postgres rather than taking it from the
    discovery stage, so a run that crashed after discovery resumes here instead of
    re-querying arXiv.
    """
    result = FetchResult()

    with db.connect() as conn:
        if arxiv_ids:
            papers = [p for p in (repo.get_paper(conn, i) for i in arxiv_ids) if p]
        else:
            papers = repo.papers_by_status(conn, "discovered", limit=limit or 500)

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})

        for paper in papers:
            arxiv_id = paper["arxiv_id"]
            pdf_path, _ = raw_paths(arxiv_id)
            result.n_attempted += 1

            # Already on disk from an earlier run - skip the network entirely.
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                with db.connect() as conn:
                    repo.mark_paper(conn, arxiv_id, "fetched", raw_path=str(pdf_path))
                result.n_cached += 1
                result.fetched_ids.append(arxiv_id)
                continue

            url = (paper.get("pdf_url") or "").replace("http://", "https://")
            if not url:
                with db.connect() as conn:
                    repo.mark_paper(conn, arxiv_id, "failed", error="no pdf_url")
                    repo.record_error(conn, run_id, "fetch", "no pdf_url", arxiv_id)
                result.n_failed += 1
                result.n_permanent += 1
                continue

            try:
                download_pdf(url, pdf_path, session=session)
            except FetchError as exc:
                with db.connect() as conn:
                    # Permanent failures stop being retried; transient ones stay at
                    # 'discovered' so the next run picks them up again.
                    if exc.permanent:
                        repo.mark_paper(conn, arxiv_id, "failed", error=str(exc))
                        result.n_permanent += 1
                    repo.record_error(conn, run_id, "fetch", str(exc), arxiv_id)
                result.n_failed += 1
                continue

            with db.connect() as conn:
                repo.mark_paper(conn, arxiv_id, "fetched", raw_path=str(pdf_path))
            result.n_fetched += 1
            result.fetched_ids.append(arxiv_id)

    return result


# --- CLI --------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(description="Download PDFs for discovered papers")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--arxiv-id", action="append",
                    help="fetch specific ids instead of the pending queue")
    args = ap.parse_args()

    result = fetch_pending(args.run_id, args.limit, args.arxiv_id)

    print(f"attempted : {result.n_attempted}")
    print(f"  fetched : {result.n_fetched}")
    print(f"  cached  : {result.n_cached}")
    print(f"  failed  : {result.n_failed} ({result.n_permanent} permanent)")

    total = 0
    for path in CONFIG.raw_dir.rglob("*.pdf"):
        total += path.stat().st_size
    print(f"\nvar/raw   : {total / 1024 / 1024:.1f} MB across "
          f"{sum(1 for _ in CONFIG.raw_dir.rglob('*.pdf'))} pdf(s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

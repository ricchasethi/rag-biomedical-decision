"""arXiv API client: date-windowed category search over the Atom export API.

Pure source adapter - it knows nothing about Postgres or Qdrant, so it can be run
and tested on its own:

    python -m biorag_pipeline.arxiv_client --days 2 --max-results 5

The arXiv API returns Atom XML. Entries are parsed into ArxivPaper, which carries
arXiv-specific fields (doi, journal_ref, comment) that the papers table does not
store; to_record() projects it onto the repository's PaperRecord. That seam is why
adding a second source later (PubMed, bioRxiv) needs no schema change.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import requests

from biorag_pipeline.config import CONFIG
from biorag_pipeline.repository import PaperRecord

ARXIV_API = "https://export.arxiv.org/api/query"

# arXiv's Terms of Use ask for no more than one request every three seconds
# from a single source, and a single connection. Both are enforced below.
MIN_REQUEST_INTERVAL = 3.0
PAGE_SIZE = 100          # arXiv caps max_results at 2000; 100 keeps responses small
MAX_RETRIES = 3
TIMEOUT = 30

USER_AGENT = "BioRAG/0.1 (biomedical decision-support RAG; +https://arxiv.org/help/api)"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


class ArxivError(RuntimeError):
    """Raised when the arXiv API itself reports a problem or stays unreachable."""


# --- Records ----------------------------------------------------------------

@dataclass
class ArxivPaper:
    """One entry from the arXiv Atom feed."""

    arxiv_id: str                       # '2509.01234' - version stripped
    version: int
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    primary_category: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    pdf_url: str | None = None
    abs_url: str | None = None
    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None

    def to_record(self) -> PaperRecord:
        """Project onto the persistence layer's PaperRecord.

        doi / journal_ref / comment are intentionally dropped - the papers table
        does not store them. Add columns first if you want them.
        """
        return PaperRecord(
            arxiv_id=self.arxiv_id,
            version=self.version,
            title=self.title,
            abstract=self.abstract,
            authors=self.authors,
            categories=self.categories,
            primary_category=self.primary_category,
            published_at=self.published_at,
            updated_at_src=self.updated_at,
            pdf_url=self.pdf_url,
        )


# --- Parsing helpers --------------------------------------------------------

def split_arxiv_id(url_or_id: str) -> tuple[str, int]:
    """Split an arXiv identifier into (base_id, version).

        http://arxiv.org/abs/2509.01234v2   -> ('2509.01234', 2)
        http://arxiv.org/abs/math.GT/0309136v1 -> ('math.GT/0309136', 1)
        2509.01234                          -> ('2509.01234', 1)

    Pre-2007 identifiers contain a slash, so the split is on the last 'v' followed
    by digits rather than on any 'v' or on '/'.
    """
    ident = url_or_id.rsplit("/abs/", 1)[-1].strip()
    base, sep, ver = ident.rpartition("v")
    if sep and ver.isdigit():
        return base, int(ver)
    return ident, 1


def _text(element: ET.Element | None) -> str:
    """Collapse arXiv's hard-wrapped text into a single normalised line."""
    if element is None or element.text is None:
        return ""
    return " ".join(element.text.split())


def _parse_datetime(raw: str) -> datetime | None:
    """Parse arXiv's RFC-3339 timestamps into timezone-aware datetimes."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_entry(entry: ET.Element) -> ArxivPaper | None:
    """Convert one <entry> element into an ArxivPaper, or None if unusable."""
    raw_id = _text(entry.find("atom:id", NS))
    if not raw_id:
        return None

    arxiv_id, version = split_arxiv_id(raw_id)

    pdf_url = None
    abs_url = None
    for link in entry.findall("atom:link", NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
        elif link.get("rel") == "alternate":
            abs_url = link.get("href")
    if pdf_url is None:
        # A small number of entries omit the pdf link; the canonical URL is
        # derivable, so construct it rather than dropping the paper.
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}v{version}"

    primary = entry.find("arxiv:primary_category", NS)

    return ArxivPaper(
        arxiv_id=arxiv_id,
        version=version,
        title=_text(entry.find("atom:title", NS)),
        abstract=_text(entry.find("atom:summary", NS)),
        authors=[
            _text(a.find("atom:name", NS))
            for a in entry.findall("atom:author", NS)
            if _text(a.find("atom:name", NS))
        ],
        categories=[
            c.get("term") for c in entry.findall("atom:category", NS) if c.get("term")
        ],
        primary_category=primary.get("term") if primary is not None else None,
        published_at=_parse_datetime(_text(entry.find("atom:published", NS))),
        updated_at=_parse_datetime(_text(entry.find("atom:updated", NS))),
        pdf_url=pdf_url,
        abs_url=abs_url,
        doi=_text(entry.find("arxiv:doi", NS)) or None,
        journal_ref=_text(entry.find("arxiv:journal_ref", NS)) or None,
        comment=_text(entry.find("arxiv:comment", NS)) or None,
    )


def parse_feed(xml_text: str) -> tuple[list[ArxivPaper], int]:
    """Parse an Atom feed into (papers, total_results).

    total_results comes from opensearch:totalResults and is how the pager knows
    when to stop. arXiv signals malformed queries with a single entry whose id
    points at /api/errors rather than with an HTTP error status, so that case is
    detected here and raised.
    """
    root = ET.fromstring(xml_text)

    total_el = root.find("opensearch:totalResults", NS)
    total = int(total_el.text) if total_el is not None and total_el.text else 0

    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", NS):
        raw_id = _text(entry.find("atom:id", NS))
        if "/api/errors" in raw_id:
            raise ArxivError(_text(entry.find("atom:summary", NS)) or "arXiv API error")
        paper = _parse_entry(entry)
        if paper is not None:
            papers.append(paper)

    return papers, total


# --- Query construction -----------------------------------------------------

def _fmt_window(moment: datetime) -> str:
    """arXiv wants submittedDate bounds as YYYYMMDDHHMM in UTC."""
    return moment.astimezone(timezone.utc).strftime("%Y%m%d%H%M")


# arXiv search field prefixes. 'all' searches title, abstract, authors and
# comments together; the others narrow to one field.
SEARCH_FIELDS = frozenset({"all", "ti", "abs", "au", "co", "jr", "cat", "rn"})


def build_query(
    categories: list[str] | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    terms: list[str] | None = None,
    field: str = "all",
) -> str:
    """Build a search_query string from up to three clauses, AND-ed together.

        categories -> (cat:q-bio.QM OR cat:q-bio.GN)      [OR within the clause]
        terms      -> (all:"protein folding" AND all:"cryo-em")
        window     -> submittedDate:[202609010000 TO 202609030000]

    Categories are OR-ed because a paper in any of them is wanted; terms are
    AND-ed because each one narrows the topic further. Terms are quoted so
    multi-word input is treated as a phrase rather than as loose keywords.

    At least one of `categories` or `terms` must be given - a bare date window
    would match all of arXiv.

    submittedDate filters on original submission, so a v2 of an old paper does not
    resurface in today's window. That is deliberate - revisions are picked up by
    re-running an older window, not by the daily one.
    """
    if field not in SEARCH_FIELDS:
        raise ValueError(f"unknown search field {field!r}; expected one of "
                         f"{sorted(SEARCH_FIELDS)}")
    if not categories and not terms:
        raise ValueError("at least one arXiv category or search term is required")

    clauses: list[str] = []

    if categories:
        clauses.append("(" + " OR ".join(f"cat:{c}" for c in categories) + ")")

    if terms:
        quoted = [f'{field}:"{t}"' for t in terms if t.strip()]
        if quoted:
            clauses.append("(" + " AND ".join(quoted) + ")")

    if window_start and window_end:
        clauses.append(
            f"submittedDate:"
            f"[{_fmt_window(window_start)} TO {_fmt_window(window_end)}]"
        )

    return " AND ".join(clauses)


# --- HTTP with throttling and retries ---------------------------------------

_last_request = 0.0
_throttle_lock = threading.Lock()


def _throttle(min_interval: float = MIN_REQUEST_INTERVAL) -> None:
    """Block until at least min_interval has passed since the previous request.

    Lock-guarded because Celery may run several mapped tasks in one process, and
    arXiv rate-limits per source, not per thread.
    """
    global _last_request
    with _throttle_lock:
        wait = min_interval - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


def _fetch_page(params: dict[str, str | int], session: requests.Session) -> str:
    """GET one page of results, retrying transient failures with backoff."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        _throttle()
        try:
            response = session.get(ARXIV_API, params=params, timeout=TIMEOUT)
            if response.status_code >= 500:
                raise ArxivError(f"arXiv returned HTTP {response.status_code}")
            response.raise_for_status()
            return response.text
        except (requests.RequestException, ArxivError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)   # 2s, 4s

    raise ArxivError(
        f"arXiv unreachable after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def search(
    categories: list[str] | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    max_results: int | None = None,
    page_size: int = PAGE_SIZE,
    terms: list[str] | None = None,
    field: str = "all",
) -> list[ArxivPaper]:
    """Return every paper matching the categories, topic terms and date window.

    Pages through the API in ascending submittedDate order - ascending because a
    descending sort shifts under you if new papers are announced mid-pagination.
    Stops at max_results, at total_results, or at the first empty page.

    Passing `terms` without `categories` searches all of arXiv for the topic;
    passing both narrows the topic to those categories. Explicitly passing
    categories=[] with terms is how you opt out of the configured default.
    """
    if categories is None and not terms:
        categories = CONFIG.arxiv_categories
    max_results = max_results or CONFIG.arxiv_max_results
    query = build_query(categories, window_start, window_end, terms, field)

    collected: list[ArxivPaper] = []
    seen: set[str] = set()
    start = 0

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})

        while len(collected) < max_results:
            params = {
                "search_query": query,
                "start": start,
                "max_results": min(page_size, max_results - len(collected)),
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
            papers, total = parse_feed(_fetch_page(params, session))

            if not papers:
                break

            for paper in papers:
                # A paper can appear twice if arXiv reindexes mid-pagination;
                # dedupe here so the caller never sees it.
                if paper.arxiv_id not in seen:
                    seen.add(paper.arxiv_id)
                    collected.append(paper)

            start += len(papers)
            if start >= total:
                break

    return collected[:max_results]


def fetch_by_ids(arxiv_ids: list[str], batch_size: int = 100) -> list[ArxivPaper]:
    """Fetch exact papers by identifier, ignoring category and date entirely.

    Uses the API's id_list parameter rather than search_query. Versions may be
    included ('2609.01055v2') to pin a specific revision; bare ids return the
    latest. Ids arXiv does not recognise are simply absent from the response, so
    compare lengths if you need to know which ones were missed.
    """
    if not arxiv_ids:
        return []

    collected: list[ArxivPaper] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        for offset in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[offset:offset + batch_size]
            params = {"id_list": ",".join(batch), "max_results": len(batch)}
            papers, _ = parse_feed(_fetch_page(params, session))
            collected.extend(papers)
    return collected


def default_window(
    lookback_days: int = 2, now: datetime | None = None
) -> tuple[datetime, datetime]:
    """The submission window a daily run should query.

    Deliberately wider than one day. arXiv's search index lags announcement, so a
    strict 24-hour window silently drops papers. The overlap costs nothing because
    upsert_paper() returns 'unchanged' for anything already stored.
    """
    end = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return end - timedelta(days=lookback_days), end


# --- CLI --------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(description="Query the arXiv API (no database writes)")
    ap.add_argument("--categories", default=",".join(CONFIG.arxiv_categories),
                    help="comma-separated arXiv categories")
    ap.add_argument("--days", type=int, default=2,
                    help="look back this many days from now (default: 2)")
    ap.add_argument("--since", help="window start, YYYY-MM-DD (overrides --days)")
    ap.add_argument("--until", help="window end, YYYY-MM-DD")
    ap.add_argument("--max-results", type=int, default=10)
    ap.add_argument("--search", action="append", metavar="PHRASE",
                    help="topic phrase; repeat to AND several together")
    ap.add_argument("--field", default="all", choices=sorted(SEARCH_FIELDS),
                    help="which field --search looks in (default: all)")
    ap.add_argument("--arxiv-id", action="append", metavar="ID",
                    help="fetch these exact ids; ignores category, topic and dates")
    ap.add_argument("--any-category", action="store_true",
                    help="search all of arXiv, not just the configured categories")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    if args.arxiv_id:
        print(f"id_list: {', '.join(args.arxiv_id)}")
        papers = fetch_by_ids(args.arxiv_id)
        missing = set(args.arxiv_id) - {p.arxiv_id for p in papers}
        if missing:
            print(f"not found: {', '.join(sorted(missing))}")
    else:
        if args.since:
            start = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
            end = (datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc)
                   if args.until else datetime.now(timezone.utc))
        else:
            start, end = default_window(args.days)

        categories = ([] if args.any_category
                      else [c.strip() for c in args.categories.split(",") if c.strip()])

        print(f"query : {build_query(categories, start, end, args.search, args.field)}")
        print(f"window: {start.isoformat()}  ->  {end.isoformat()}")

        papers = search(categories, start, end, max_results=args.max_results,
                        terms=args.search, field=args.field)

    if args.json:
        print(json.dumps([asdict(p) for p in papers], default=str, indent=2))
        return 0

    print(f"\n{len(papers)} paper(s)\n")
    for p in papers:
        published = p.published_at.date().isoformat() if p.published_at else "?"
        print(f"  {p.arxiv_id:<18} v{p.version}  {published}  "
              f"[{p.primary_category}]")
        print(f"    {p.title[:88]}")
        print(f"    {len(p.authors)} author(s) | {p.pdf_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

"""Offline tests for the arXiv client.

No network. Every assertion runs against a fixture Atom feed, so these are fast,
deterministic, and safe in CI. The live API is exercised separately by the CLI:

    python -m biorag_pipeline.arxiv_client --days 3 --max-results 5

Run:
    docker compose exec airflow-worker python /opt/biorag/tests/test_arxiv_client.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from biorag_pipeline.arxiv_client import (
    ArxivError,
    build_query,
    default_window,
    parse_feed,
    split_arxiv_id,
)

passed = 0


def check(label: str, condition: bool) -> None:
    global passed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        raise AssertionError(label)


# --- Fixtures ---------------------------------------------------------------
# Entry 1: modern id, v2, hard-wrapped title and summary, pdf link present.
# Entry 2: pre-2007 id containing a slash, and NO pdf link, to exercise the
#          fallback URL construction.

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>2</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <entry>
    <id>http://arxiv.org/abs/2509.01234v2</id>
    <updated>2026-09-02T10:00:00Z</updated>
    <published>2026-09-01T17:59:59Z</published>
    <title>A Deep Learning Approach to
  Protein Folding</title>
    <summary>  We present a method.
  It works well.
</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:primary_category term="q-bio.QM"/>
    <category term="q-bio.QM" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2509.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2509.01234v2" rel="related"
          type="application/pdf"/>
    <arxiv:doi>10.1234/example</arxiv:doi>
    <arxiv:comment>12 pages, 3 figures</arxiv:comment>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/math.GT/0309136v1</id>
    <updated>2003-09-09T12:00:00Z</updated>
    <published>2003-09-09T12:00:00Z</published>
    <title>An Old Paper</title>
    <summary>Older identifier scheme.</summary>
    <author><name>Grigori Perelman</name></author>
    <arxiv:primary_category term="math.GT"/>
    <category term="math.GT" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""

# arXiv reports malformed queries with HTTP 200 and an entry pointing at /api/errors.
ERROR_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format</id>
    <title>Error</title>
    <summary>incorrect id format for nonsense</summary>
  </entry>
</feed>
"""


def main() -> int:
    print("\n-- identifier splitting --")
    check("modern id with version",
          split_arxiv_id("http://arxiv.org/abs/2509.01234v2") == ("2509.01234", 2))
    check("pre-2007 id with a slash",
          split_arxiv_id("http://arxiv.org/abs/math.GT/0309136v1")
          == ("math.GT/0309136", 1))
    check("bare id, no URL prefix",
          split_arxiv_id("2509.01234v3") == ("2509.01234", 3))
    check("no version defaults to 1",
          split_arxiv_id("2509.01234") == ("2509.01234", 1))
    check("trailing 'v' that is not a version",
          split_arxiv_id("2509.01234vX") == ("2509.01234vX", 1))

    print("\n-- feed parsing --")
    papers, total = parse_feed(FEED)
    check("both entries parsed", len(papers) == 2)
    check("totalResults read", total == 2)

    first, second = papers
    check("version extracted from id", first.version == 2)
    check("id stripped of version", first.arxiv_id == "2509.01234")
    check("hard-wrapped title normalised",
          first.title == "A Deep Learning Approach to Protein Folding")
    check("summary whitespace collapsed",
          first.abstract == "We present a method. It works well.")
    check("authors in order",
          first.authors == ["Ada Lovelace", "Alan Turing"])
    check("all categories captured",
          first.categories == ["q-bio.QM", "cs.LG"])
    check("primary category is the arxiv: one",
          first.primary_category == "q-bio.QM")
    check("published_at is timezone-aware",
          first.published_at == datetime(2026, 9, 1, 17, 59, 59, tzinfo=timezone.utc))
    check("updated_at parsed",
          first.updated_at == datetime(2026, 9, 2, 10, 0, 0, tzinfo=timezone.utc))
    check("pdf url from the link element",
          first.pdf_url == "http://arxiv.org/pdf/2509.01234v2")
    check("arxiv-only fields captured", first.doi == "10.1234/example")

    print("\n-- edge cases --")
    check("old-style id survives parsing", second.arxiv_id == "math.GT/0309136")
    check("missing pdf link falls back to a constructed url",
          second.pdf_url == "https://arxiv.org/pdf/math.GT/0309136v1")
    check("absent optional fields are None", second.doi is None)

    print("\n-- projection onto PaperRecord --")
    record = first.to_record()
    check("arxiv_id carried over", record.arxiv_id == "2509.01234")
    check("version carried over", record.version == 2)
    check("updated_at maps to updated_at_src",
          record.updated_at_src == first.updated_at)
    check("PaperRecord has no doi field", not hasattr(record, "doi"))

    print("\n-- query construction --")
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 3, tzinfo=timezone.utc)
    check("categories OR-ed and window AND-ed",
          build_query(["q-bio.QM", "q-bio.GN"], start, end)
          == "(cat:q-bio.QM OR cat:q-bio.GN) AND "
             "submittedDate:[202609010000 TO 202609030000]")
    check("no window means no date clause",
          build_query(["q-bio.QM"]) == "(cat:q-bio.QM)")

    try:
        build_query([])
        check("empty categories rejected", False)
    except ValueError:
        check("empty categories rejected", True)

    print("\n-- error handling --")
    try:
        parse_feed(ERROR_FEED)
        check("HTTP-200 error feed raises", False)
    except ArxivError as exc:
        check("HTTP-200 error feed raises", "incorrect id format" in str(exc))

    print("\n-- default window --")
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    w_start, w_end = default_window(lookback_days=2, now=now)
    check("window ends at now", w_end == now)
    check("window spans the lookback", (w_end - w_start).days == 2)
    check("window bounds are UTC-aware", w_start.tzinfo is not None)

    print(f"\n{passed} checks passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

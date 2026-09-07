"""Parse stage: PDF -> clean sectioned text.

Third stage of the DAG. Reads papers with status='fetched', extracts text with
pypdf, writes it next to the PDF as .txt, and advances them to 'parsed'.

    python -m biorag_pipeline.parse --limit 5
    python -m biorag_pipeline.parse --arxiv-id 2609.01055 --show

Why the extracted text is written to disk rather than kept in memory or in a
Postgres column: re-chunking with a different chunk_size then costs nothing - no
re-download, no re-extraction. That was the point of the three-tier storage split.

Two integration details with core/rag_engine.py, both easy to get wrong:

* TextProcessor.clean_text() already de-hyphenates line breaks ('bio- marker' ->
  'biomarker'), so this module must NOT do that. Doing it twice mangles real
  hyphenated terms.

* DocumentChunker._detect_section() runs on *sentences*, after clean_text has
  collapsed every newline. A bare 'Introduction' heading therefore only registers
  if it begins a sentence, so headings are emitted here as 'Introduction.' with a
  terminating period. Without it every chunk silently ends up in section 'Body'
  and the reranker's SECTION_WEIGHTS never fire.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

from biorag_pipeline import db, repository as repo
from biorag_pipeline.fetch import raw_paths

# Below this many characters the extraction is treated as failed (scanned or
# image-only PDF) and the abstract is used instead.
MIN_USABLE_CHARS = 800

# Headings are only accepted when the whole line is the heading, optionally
# preceded by a section number. That keeps 'we discuss the results below' from
# being mistaken for a Results heading.
_HEADING_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*|[IVXLC]+)\s*[.)]?\s+)?"
    r"([A-Za-z][A-Za-z \-]{2,40}?)\s*$"
)

# Canonical names match DocumentChunker._detect_section's patterns exactly.
# None means the section is dropped entirely.
_CANONICAL: dict[str, str | None] = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Introduction",
    "related work": "Introduction",
    "method": "Methods",
    "methods": "Methods",
    "methodology": "Methods",
    "materials and methods": "Methods",
    "method and materials": "Methods",
    "result": "Results",
    "results": "Results",
    "results and discussion": "Results",
    "experiments": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "conclusions and future work": "Conclusion",
    "reference": "References",
    "references": "References",
    "bibliography": "References",
    "appendix": "Supplementary",
    "supplementary": "Supplementary",
    "supplementary material": "Supplementary",
    "supplementary materials": "Supplementary",
    "acknowledgment": None,
    "acknowledgments": None,
    "acknowledgement": None,
    "acknowledgements": None,
    "funding": None,
    "competing interests": None,
    "conflict of interest": None,
}

# arXiv stamps the left margin of page 1; pypdf extracts it as a stray line.
_ARXIV_STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[[\w.\-]+\]\s*\d+\s+\w+\s+\d{4}")
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*(?:of\s+\d{1,4})?\s*$", re.I)


# Control characters that PDF extraction emits but text can never legitimately
# contain. Tab (09), newline (0a) and carriage return (0d) are deliberately kept.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str) -> str:
    """Remove control characters that would poison downstream storage.

    NUL (0x00) is the one that actually breaks things: a Postgres text column
    cannot hold it, and psycopg2 raises ValueError before the statement is even
    sent - so a single such byte fails an entire batch insert. pypdf produces them
    for certain embedded font encodings, so this is data, not a bug.

    Applied before hashing, so content_hash describes the text that is actually
    stored rather than the raw extraction.
    """
    return _CONTROL_CHARS.sub("", text)


@dataclass
class ParsedDocument:
    """Extracted text plus what the pipeline needs to decide whether to re-index."""

    text: str
    content_hash: str
    n_pages: int
    n_chars: int
    sections: list[str] = field(default_factory=list)
    source: Literal["pdf", "abstract"] = "pdf"


@dataclass
class ParseResult:
    n_attempted: int = 0
    n_parsed: int = 0
    n_unchanged: int = 0       # same content_hash and already chunked - skipped
    n_abstract_only: int = 0
    n_failed: int = 0
    parsed_ids: list[str] = field(default_factory=list)


# --- Cleaning helpers -------------------------------------------------------

def _drop_running_heads(pages: list[str]) -> list[str]:
    """Remove journal headers, footers and page numbers repeated across pages.

    Any short line that appears at the top or bottom of most pages is furniture,
    not content. Left in, it pollutes every chunk and skews BM25 term statistics.
    """
    if len(pages) < 3:
        return pages

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        for line in lines[:2] + lines[-2:]:
            counts[line] += 1

    threshold = max(2, int(len(pages) * 0.6))
    furniture = {
        line for line, count in counts.items()
        if count >= threshold and len(line) < 120
    }

    cleaned = []
    for page in pages:
        kept = [
            line for line in page.splitlines()
            if line.strip() not in furniture
            and not _PAGE_NUMBER.match(line)
            and not _ARXIV_STAMP.search(line)
        ]
        cleaned.append("\n".join(kept))
    return cleaned


def _canonical_heading(line: str) -> str | None | Literal[False]:
    """Return the canonical section name, None to drop the line, or False if the
    line is not a heading at all."""
    match = _HEADING_RE.match(line)
    if not match:
        return False
    key = " ".join(match.group(1).split()).lower()
    if key not in _CANONICAL:
        return False
    return _CANONICAL[key]


def normalise_sections(text: str, drop_references: bool = True) -> tuple[str, list[str]]:
    """Rewrite recognised headings into the engine's canonical, sentence-terminated
    form and optionally truncate at the reference list.

    Reference lists are typically a third of a paper and are pure lexical noise:
    author names and title fragments that match almost any query under BM25 while
    answering none of them. Dropping them is the single largest retrieval-quality
    win available at parse time.
    """
    output: list[str] = []
    seen: list[str] = []
    dropping = False

    for line in text.splitlines():
        heading = _canonical_heading(line)

        if heading is not False:
            if heading is None:
                dropping = True          # acknowledgements, funding, ...
                continue
            if heading == "References" and drop_references:
                break
            dropping = False
            if not seen or seen[-1] != heading:
                seen.append(heading)
            output.append(f"{heading}.")
            continue

        if not dropping:
            output.append(line)

    return "\n".join(output), seen


# --- Extraction -------------------------------------------------------------

def parse_pdf(path: Path, drop_references: bool = True) -> ParsedDocument:
    """Extract text from a PDF and normalise its structure."""
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    pages = _drop_running_heads(pages)

    raw = sanitize_text("\n".join(pages))
    text, sections = normalise_sections(raw, drop_references)

    # Collapse runs of blank lines but keep single newlines: clean_text() in the
    # engine flattens them anyway, and keeping them makes the .txt readable.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return ParsedDocument(
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        n_pages=len(reader.pages),
        n_chars=len(text),
        sections=sections,
        source="pdf",
    )


def from_abstract(title: str, abstract: str) -> ParsedDocument:
    """Fallback document built from metadata when a PDF yields no usable text.

    Mirrors ingestion_pubmed.py's abstract-only path: a scanned or image-only PDF
    still contributes a searchable record rather than being dropped.
    """
    text = sanitize_text(f"{title}\nAbstract.\n{abstract}".strip())
    return ParsedDocument(
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        n_pages=0,
        n_chars=len(text),
        sections=["Abstract"],
        source="abstract",
    )


# --- Stage ------------------------------------------------------------------

def parse_pending(run_id: str | None = None, limit: int | None = None,
                  arxiv_ids: list[str] | None = None,
                  drop_references: bool = True) -> ParseResult:
    """Parse every paper at status='fetched' and write its text beside the PDF."""
    result = ParseResult()

    with db.connect() as conn:
        if arxiv_ids:
            papers = [p for p in (repo.get_paper(conn, i) for i in arxiv_ids) if p]
        else:
            papers = repo.papers_by_status(conn, "fetched", limit=limit or 500)

    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        pdf_path, text_path = raw_paths(arxiv_id)
        result.n_attempted += 1

        try:
            document = parse_pdf(pdf_path, drop_references)
        except Exception as exc:
            document = None
            with db.connect() as conn:
                repo.record_error(conn, run_id, "parse",
                                  f"pdf extraction failed: {exc}", arxiv_id)

        if document is None or document.n_chars < MIN_USABLE_CHARS:
            abstract = paper.get("abstract") or ""
            if not abstract:
                with db.connect() as conn:
                    repo.mark_paper(conn, arxiv_id, "failed",
                                    error="no usable text and no abstract")
                    repo.record_error(conn, run_id, "parse",
                                      "no usable text and no abstract", arxiv_id)
                result.n_failed += 1
                continue
            if document is not None:
                with db.connect() as conn:
                    repo.record_error(
                        conn, run_id, "parse",
                        f"only {document.n_chars} chars extracted; "
                        "falling back to abstract", arxiv_id)
            document = from_abstract(paper["title"], abstract)
            result.n_abstract_only += 1

        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(document.text, encoding="utf-8")

        with db.connect() as conn:
            changed = repo.content_changed(conn, arxiv_id, document.content_hash)
            if not changed and (paper.get("n_chunks") or 0) > 0:
                # Same text as last time and chunks already exist: skip straight to
                # 'indexed'. This is what makes an arXiv v2 that only fixed an
                # author affiliation cost nothing to re-ingest.
                repo.mark_paper(conn, arxiv_id, "indexed")
                result.n_unchanged += 1
                continue
            repo.mark_paper(conn, arxiv_id, "parsed",
                            content_hash=document.content_hash)

        result.n_parsed += 1
        result.parsed_ids.append(arxiv_id)

    return result


# --- CLI --------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(description="Extract text from fetched PDFs")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--arxiv-id", action="append",
                    help="parse specific ids instead of the pending queue")
    ap.add_argument("--keep-references", action="store_true",
                    help="do not truncate at the reference list")
    ap.add_argument("--show", action="store_true",
                    help="print the first 1500 characters of each parsed document")
    args = ap.parse_args()

    result = parse_pending(args.run_id, args.limit, args.arxiv_id,
                           drop_references=not args.keep_references)

    print(f"attempted    : {result.n_attempted}")
    print(f"  parsed     : {result.n_parsed}")
    print(f"  unchanged  : {result.n_unchanged}")
    print(f"  abstract   : {result.n_abstract_only}")
    print(f"  failed     : {result.n_failed}")

    if args.show:
        for arxiv_id in result.parsed_ids[:3]:
            _, text_path = raw_paths(arxiv_id)
            print(f"\n{'=' * 70}\n{arxiv_id}\n{'=' * 70}")
            print(text_path.read_text(encoding="utf-8")[:1500])
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

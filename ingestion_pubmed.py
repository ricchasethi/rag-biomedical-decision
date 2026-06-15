import requests
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from core.rag_engine import BioRAGEngine

_CORPUS_PATH = Path(__file__).parent / "data" / "sample_corpus.py"


@dataclass
class IngestionError:
    """A single item that was skipped during ingestion, with its cause."""
    stage: str       # "elink", "pmc_fetch", "pmc_parse", "pubmed_fetch", "pubmed_parse"
    identifier: str  # PMID, PMCID, positional index, or "batch"
    reason: str      # str(exception)


@dataclass
class IngestionResult:
    """Structured return value from ingest_pubmed."""
    engine: BioRAGEngine
    indexed: int
    errors: list[IngestionError] = field(default_factory=list)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EMAIL = "ricchasethi@gmail.com"   # NCBI requires this
TOOL = "biorag-ingestor"

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# ─────────────────────────────────────────────
# PUBMED CLIENT
# ─────────────────────────────────────────────

def search_pubmed(query: str, max_results: int = 10) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    url = BASE_URL + "esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "tool": TOOL,
        "email": EMAIL,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    return data["esearchresult"]["idlist"]


def fetch_pubmed_details(pmids: list[str]) -> str:
    """Fetch abstracts + metadata for given PMIDs."""
    url = BASE_URL + "efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": TOOL,
        "email": EMAIL,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.text


def get_pmc_ids_for_pmids(
    pmids: list[str],
) -> tuple[dict[str, str], list[IngestionError]]:
    """Use elink to find PMC IDs linked to given PubMed IDs.

    Returns a dict mapping PMID → PMCID for articles available in PMC, plus
    a list of per-PMID errors for any elink requests that failed.
    PMIDs without a PMC full-text record are omitted from the mapping.

    One request is made per PMID because batching collapses all results
    into a single linkset with no per-PMID attribution, making it
    impossible to reconstruct the mapping from a single batched call.
    """
    url = BASE_URL + "elink.fcgi"
    pmid_to_pmcid: dict[str, str] = {}
    errors: list[IngestionError] = []

    for pmid in pmids:
        params = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "json",
            "tool": TOOL,
            "email": EMAIL,
        }
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()

            for linkset in data.get("linksets", []):
                for linksetdb in linkset.get("linksetdbs", []):
                    if linksetdb.get("dbto") == "pmc":
                        pmc_links = linksetdb.get("links", [])
                        if pmc_links:
                            pmid_to_pmcid[pmid] = pmc_links[0]
        except Exception as exc:
            errors.append(IngestionError(stage="elink", identifier=pmid, reason=str(exc)))

        time.sleep(0.34)  # NCBI rate limit: ≤3 requests/second without API key

    return pmid_to_pmcid, errors


def fetch_pmc_fulltext(pmcids: list[str]) -> str:
    """Fetch full-text JATS XML from PubMed Central for the given PMC IDs."""
    url = BASE_URL + "efetch.fcgi"
    params = {
        "db": "pmc",
        "id": ",".join(pmcids),
        "retmode": "xml",
        "tool": TOOL,
        "email": EMAIL,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.text


# ─────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────

def _iter_text(element: ET.Element) -> str:
    """Concatenate all text nodes within an XML element (including tails)."""
    return "".join(element.itertext()).strip()


def _strip_citation_xrefs(root: ET.Element) -> None:
    """Remove inline citation markers from a JATS XML tree in-place.

    In JATS XML, every in-text citation appears as:
        <xref ref-type="bibr" rid="bib1">1</xref>
    When itertext() concatenates the surrounding text, the citation number
    gets stitched directly onto the preceding word, producing artifacts like:
        "DKD.1 Within"   "fibrosis.34,35 However"   "therapy,3 hypoglycemic"

    This function removes those xref elements while correctly preserving the
    tail text (the text that follows the closing </xref> tag in the source),
    which belongs to the sentence and must not be lost.

    Why tail text matters in ElementTree:
        <p>DKD<xref>1</xref> Within China</p>
        element.text  = "DKD"
        xref.text     = "1"
        xref.tail     = " Within China"   ← this lives on the xref, not the <p>
    Removing the xref without splicing the tail would silently drop " Within China".
    """
    for parent in root.iter():
        to_remove = [
            (i, child)
            for i, child in enumerate(parent)
            if child.tag == "xref" and child.get("ref-type") == "bibr"
        ]
        # Iterate in reverse so indices stay valid as we remove elements
        for i, child in reversed(to_remove):
            tail = child.tail or ""
            siblings = list(parent)
            if i == 0:
                # No preceding sibling: splice tail onto the parent's own text
                parent.text = (parent.text or "") + tail
            else:
                # Splice tail onto the preceding sibling's tail
                prev = siblings[i - 1]
                prev.tail = (prev.tail or "") + tail
            parent.remove(child)


def parse_pubmed_xml(xml_text: str) -> tuple[list[dict], list[IngestionError]]:
    """Extract structured papers (abstract only) from PubMed efetch XML."""
    root = ET.fromstring(xml_text)
    papers: list[dict] = []
    errors: list[IngestionError] = []

    for i, article in enumerate(root.findall(".//PubmedArticle")):
        identifier = f"article_{i}"
        try:
            pmid = article.findtext(".//PMID") or identifier
            identifier = pmid

            title = article.findtext(".//ArticleTitle", default="No title")
            abstract_parts = article.findall(".//AbstractText")
            abstract = "\n".join([a.text or "" for a in abstract_parts])
            year = article.findtext(".//PubDate/Year", default="Unknown")
            journal = article.findtext(".//Journal/Title", default="Unknown")
            text = f"Abstract\n{abstract}".strip()

            papers.append({
                "id": f"pmid_{pmid}",
                "title": title,
                "text": text,
                "metadata": {
                    "year": year,
                    "journal": journal,
                    "source": "pubmed",
                    "pmid": pmid,
                    "has_full_text": False,
                },
            })
        except Exception as exc:
            errors.append(IngestionError(stage="pubmed_parse", identifier=identifier, reason=str(exc)))

    return papers, errors


def parse_pmc_xml(xml_text: str) -> tuple[list[dict], list[IngestionError]]:
    """Extract structured full-text papers from PMC JATS XML.

    Parses the <body> element for section titles and paragraphs.
    Falls back to the abstract when no body is present (e.g. open-access
    metadata-only records).
    """
    root = ET.fromstring(xml_text)
    papers: list[dict] = []
    errors: list[IngestionError] = []

    for i, article in enumerate(root.findall(".//article")):
        identifier = f"article_{i}"
        try:
            # Strip inline citation markers before any text extraction.
            # Must happen first so _iter_text() never sees the raw citation numbers.
            _strip_citation_xrefs(article)

            pmcid = article.findtext(".//article-id[@pub-id-type='pmc']") or ""
            pmid  = article.findtext(".//article-id[@pub-id-type='pmid']") or ""
            identifier = f"pmc_{pmcid}" if pmcid else f"pmid_{pmid}" if pmid else identifier

            title = article.findtext(".//article-title", default="No title")

            # Year — try multiple pub-date variants
            year = (
                article.findtext(".//pub-date[@pub-type='epub']/year")
                or article.findtext(".//pub-date[@pub-type='ppub']/year")
                or article.findtext(".//pub-date/year")
                or "Unknown"
            )
            journal = article.findtext(".//journal-title", default="Unknown")

            # Abstract
            abstract_paras = article.findall(".//abstract//p")
            abstract_text = "\n".join(
                _iter_text(p) for p in abstract_paras if _iter_text(p)
            )
            # Fallback: abstract without <p> wrappers
            if not abstract_text:
                for ab in article.findall(".//abstract"):
                    raw = _iter_text(ab)
                    if raw:
                        abstract_text = raw
                        break

            # Full-text body sections
            sections: list[str] = []
            body = article.find(".//body")
            if body is not None:
                for sec in body.iter("sec"):
                    sec_title = sec.findtext("title", default="").strip()
                    # Collect only direct-child <p> elements to avoid duplication
                    # from nested subsections (they will appear in their own sec iter).
                    paras = [
                        _iter_text(p)
                        for p in sec
                        if p.tag == "p" and _iter_text(p)
                    ]
                    if paras:
                        header = f"\n{sec_title}" if sec_title else ""
                        sections.append(header + "\n" + "\n".join(paras))

            # Assemble document text
            text_parts: list[str] = []
            if abstract_text:
                text_parts.append(f"Abstract\n{abstract_text}")
            text_parts.extend(sections)

            text = "\n\n".join(text_parts).strip()
            if not text:
                continue

            # Prefer PMC ID; fall back to PMID so every article gets a unique key
            doc_id = f"pmc_{pmcid}" if pmcid else f"pmid_{pmid}" if pmid else f"pmc_unknown_{hash(title) & 0xFFFFFF}"
            papers.append({
                "id": doc_id,
                "title": title,
                "text": text,
                "metadata": {
                    "year": year,
                    "journal": journal,
                    "source": "pubmed_central",
                    "pmcid": pmcid,
                    "pmid": pmid,
                    "has_full_text": bool(sections),
                },
            })
        except Exception as exc:
            errors.append(IngestionError(stage="pmc_parse", identifier=identifier, reason=str(exc)))

    return papers, errors


# ─────────────────────────────────────────────
# CORPUS PERSISTENCE
# ─────────────────────────────────────────────

def save_to_corpus(
    papers: list[dict],
    corpus_path: Path | None = None,
) -> int:
    """Append new papers to data/sample_corpus.py.

    Each paper dict must have the same shape that ingest_pubmed produces:
    {"id", "title", "text", "metadata"}.

    Papers whose doc_id already exists in SAMPLE_DOCUMENTS are skipped so
    re-running ingestion on the same query is always safe.

    Returns the number of entries actually written.

    How it works
    ────────────
    The corpus file ends with  },\n\n]  (the closing bracket of
    SAMPLE_DOCUMENTS).  We strip that trailing bracket, append each new
    entry formatted as a Python dict literal, then close the list again.
    Using repr() for all string values safely escapes quotes, backslashes,
    and any Unicode characters that appear in biomedical titles and text.
    """
    if corpus_path is None:
        corpus_path = _CORPUS_PATH

    # Discover existing IDs without importing the whole module again (avoids
    # stale-cache issues when this function is called from the same process
    # that already imported sample_corpus earlier).
    from data.sample_corpus import SAMPLE_DOCUMENTS
    existing_ids: set[str] = {d["id"] for d in SAMPLE_DOCUMENTS}

    new_papers = [p for p in papers if p["id"] not in existing_ids]
    if not new_papers:
        return 0

    raw = corpus_path.read_text(encoding="utf-8")

    # Strip the closing bracket (last ']') and any trailing whitespace so we
    # can insert entries before it.
    close_pos = raw.rfind("]")
    if close_pos == -1:
        raise ValueError(f"Could not find closing ']' in {corpus_path}")
    preamble = raw[:close_pos].rstrip()

    entries: list[str] = []
    for paper in new_papers:
        meta = paper.get("metadata") or {}
        # Build metadata lines individually so booleans stay as Python literals
        # (True/False) rather than repr strings.
        meta_lines = [
            f'            "year": {repr(meta.get("year", "Unknown"))},',
            f'            "journal": {repr(meta.get("journal", "Unknown"))},',
            f'            "source": {repr(meta.get("source", "pubmed"))},',
            f'            "pmcid": {repr(meta.get("pmcid", ""))},',
            f'            "pmid": {repr(meta.get("pmid", ""))},',
            f'            "has_full_text": {meta.get("has_full_text", False)},',
        ]
        entry = (
            "    {\n"
            f'        "id": {repr(paper["id"])},\n'
            f'        "title": {repr(paper["title"])},\n'
            f'        "text": {repr(paper["text"])},\n'
            "        \"metadata\": {\n"
            + "\n".join(meta_lines) + "\n"
            "        },\n"
            "    },"
        )
        entries.append(entry)

    new_content = preamble + "\n\n" + "\n\n".join(entries) + "\n\n]\n"
    corpus_path.write_text(new_content, encoding="utf-8")
    return len(new_papers)


# ─────────────────────────────────────────────
# INGESTION PIPELINE
# ─────────────────────────────────────────────

def ingest_pubmed(
    query: str,
    max_results: int = 10,
    engine: BioRAGEngine | None = None,
    save_corpus: bool = False,
    dense_retriever: "DenseRetriever | None" = None,
) -> IngestionResult:
    """Ingest papers for *query* into a BioRAGEngine.

    For each paper found on PubMed, the pipeline first attempts to retrieve
    the full text from PubMed Central (PMC).  Articles not available in PMC
    fall back to their PubMed abstract.

    If *engine* is supplied, documents are added to it directly (useful for
    the REST server or CLI where an existing corpus should be augmented).
    Otherwise a fresh BioRAGEngine is created and returned.  When *engine* is
    not supplied, *dense_retriever* (if given) is wired into the new engine so
    ingested chunks flow to both the BM25 and dense indexes via add_document().

    When *save_corpus* is True, all successfully indexed papers are appended
    to data/sample_corpus.py so they persist across process restarts.
    Papers already present in the corpus are silently skipped.

    Returns an IngestionResult with the engine, the count of successfully
    indexed papers, and a list of IngestionErrors for any items that were
    skipped (parse failures, per-PMID network errors, batch fetch errors).
    """
    if engine is None:
        engine = BioRAGEngine(dense_retriever=dense_retriever)

    all_errors: list[IngestionError] = []

    print(f"Searching PubMed for: {query}")
    pmids = search_pubmed(query, max_results)
    print(f"Found {len(pmids)} PubMed IDs")

    if not pmids:
        print("No results found.")
        return IngestionResult(engine=engine, indexed=0)

    # ── Step 1: resolve which PMIDs have PMC full text ───────────────────────
    print("Resolving PMC full-text availability via elink...")
    pmid_to_pmcid, link_errors = get_pmc_ids_for_pmids(pmids)
    all_errors.extend(link_errors)
    pmcids = list(pmid_to_pmcid.values())
    pmids_without_pmc = [p for p in pmids if p not in pmid_to_pmcid]

    print(f"  Full text available (PMC): {len(pmcids)}")
    print(f"  Abstract only (PubMed):    {len(pmids_without_pmc)}")

    # ── Step 2: fetch + index PMC full-text articles ─────────────────────────
    pmc_papers: list[dict] = []
    if pmcids:
        time.sleep(0.34)
        try:
            pmc_xml = fetch_pmc_fulltext(pmcids)
            pmc_papers, parse_errors = parse_pmc_xml(pmc_xml)
            all_errors.extend(parse_errors)
        except Exception as exc:
            all_errors.append(IngestionError(stage="pmc_fetch", identifier="batch", reason=str(exc)))

    for paper in pmc_papers:
        n = engine.add_document(
            doc_id=paper["id"],
            title=paper["title"],
            text=paper["text"],
            metadata=paper["metadata"],
        )
        ft_label = "full text" if paper["metadata"].get("has_full_text") else "abstract"
        print(f"  [PMC/{ft_label}] {paper['title'][:60]}... ({n} chunks)")
        time.sleep(0.1)

    # ── Step 3: fetch + index abstract-only articles ─────────────────────────
    abstract_papers: list[dict] = []
    if pmids_without_pmc:
        time.sleep(0.34)
        try:
            pubmed_xml = fetch_pubmed_details(pmids_without_pmc)
            abstract_papers, parse_errors = parse_pubmed_xml(pubmed_xml)
            all_errors.extend(parse_errors)
        except Exception as exc:
            all_errors.append(IngestionError(stage="pubmed_fetch", identifier="batch", reason=str(exc)))

    for paper in abstract_papers:
        n = engine.add_document(
            doc_id=paper["id"],
            title=paper["title"],
            text=paper["text"],
            metadata=paper["metadata"],
        )
        print(f"  [PubMed/abstract] {paper['title'][:60]}... ({n} chunks)")
        time.sleep(0.1)

    total = len(pmc_papers) + len(abstract_papers)
    print(f"\nIngestion complete — {total} papers indexed, {len(all_errors)} skipped")
    print(engine.get_corpus_stats())

    if save_corpus:
        all_papers = pmc_papers + abstract_papers
        saved = save_to_corpus(all_papers)
        print(f"Saved {saved} new paper(s) to data/sample_corpus.py")

    return IngestionResult(engine=engine, indexed=total, errors=all_errors)


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PubMed papers into BioRAG")
    parser.add_argument("--query", default="alzheimer's disease biomarkers",
                        help="PubMed search query")
    parser.add_argument("--max-results", type=int, default=5,
                        help="Maximum number of papers to fetch (default: 5)")
    parser.add_argument("--save-corpus", action="store_true",
                        help="Append ingested papers to data/sample_corpus.py")
    parser.add_argument("--hybrid", action="store_true",
                        help="Enable hybrid BM25 + dense retrieval via Qdrant")
    args = parser.parse_args()

    dense_retriever = None
    if args.hybrid:
        from hybrid_retrieval import EmbeddingModel, DenseRetriever
        dense_retriever = DenseRetriever(EmbeddingModel())

    ingest_result = ingest_pubmed(
        query=args.query,
        max_results=args.max_results,
        save_corpus=args.save_corpus,
        dense_retriever=dense_retriever,
    )
    if ingest_result.errors:
        print(f"\nSkipped {len(ingest_result.errors)} item(s):")
        for err in ingest_result.errors:
            print(f"  [{err.stage}/{err.identifier}] {err.reason}")

    result = ingest_result.engine.query("What biomarkers predict alzheimer's disease?")
    print("\n--- ANSWER ---\n")
    print(result.answer)

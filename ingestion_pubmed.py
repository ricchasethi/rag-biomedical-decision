import requests
import time
import xml.etree.ElementTree as ET
from core.rag_engine import BioRAGEngine

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


def get_pmc_ids_for_pmids(pmids: list[str]) -> dict[str, str]:
    """Use elink to find PMC IDs linked to given PubMed IDs.

    Returns a dict mapping PMID → PMCID for articles available in PMC.
    PMIDs without a PMC full-text record are omitted from the result.

    One request is made per PMID because batching collapses all results
    into a single linkset with no per-PMID attribution, making it
    impossible to reconstruct the mapping from a single batched call.
    """
    url = BASE_URL + "elink.fcgi"
    pmid_to_pmcid: dict[str, str] = {}

    for pmid in pmids:
        params = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": pmid,
            "retmode": "json",
            "tool": TOOL,
            "email": EMAIL,
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        for linkset in data.get("linksets", []):
            for linksetdb in linkset.get("linksetdbs", []):
                if linksetdb.get("dbto") == "pmc":
                    pmc_links = linksetdb.get("links", [])
                    if pmc_links:
                        pmid_to_pmcid[pmid] = pmc_links[0]

        time.sleep(0.34)  # NCBI rate limit: ≤3 requests/second without API key

    return pmid_to_pmcid


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


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Extract structured papers (abstract only) from PubMed efetch XML."""
    root = ET.fromstring(xml_text)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        try:
            title = article.findtext(".//ArticleTitle", default="No title")

            abstract_parts = article.findall(".//AbstractText")
            abstract = "\n".join([a.text or "" for a in abstract_parts])

            year = article.findtext(".//PubDate/Year", default="Unknown")
            journal = article.findtext(".//Journal/Title", default="Unknown")
            pmid = article.findtext(".//PMID", default="unknown")

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

        except Exception as e:
            print(f"Skipping PubMed article due to error: {e}")

    return papers


def parse_pmc_xml(xml_text: str) -> list[dict]:
    """Extract structured full-text papers from PMC JATS XML.

    Parses the <body> element for section titles and paragraphs.
    Falls back to the abstract when no body is present (e.g. open-access
    metadata-only records).
    """
    root = ET.fromstring(xml_text)
    papers = []

    for article in root.findall(".//article"):
        try:
            pmcid = article.findtext(".//article-id[@pub-id-type='pmc']") or ""
            pmid  = article.findtext(".//article-id[@pub-id-type='pmid']") or ""
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

        except Exception as e:
            print(f"Skipping PMC article due to error: {e}")

    return papers


# ─────────────────────────────────────────────
# INGESTION PIPELINE
# ─────────────────────────────────────────────

def ingest_pubmed(
    query: str,
    max_results: int = 10,
    engine: BioRAGEngine | None = None,
) -> BioRAGEngine:
    """Ingest papers for *query* into a BioRAGEngine.

    For each paper found on PubMed, the pipeline first attempts to retrieve
    the full text from PubMed Central (PMC).  Articles not available in PMC
    fall back to their PubMed abstract.

    If *engine* is supplied, documents are added to it directly (useful for
    the REST server or CLI where an existing corpus should be augmented).
    Otherwise a fresh BioRAGEngine is created and returned.
    """
    if engine is None:
        engine = BioRAGEngine()

    print(f"Searching PubMed for: {query}")
    pmids = search_pubmed(query, max_results)
    print(f"Found {len(pmids)} PubMed IDs")

    if not pmids:
        print("No results found.")
        return engine

    # ── Step 1: resolve which PMIDs have PMC full text ───────────────────────
    print("Resolving PMC full-text availability via elink...")
    pmid_to_pmcid = get_pmc_ids_for_pmids(pmids)  # sleep is inside, one req/PMID
    pmcids = list(pmid_to_pmcid.values())
    pmids_without_pmc = [p for p in pmids if p not in pmid_to_pmcid]

    print(f"  Full text available (PMC): {len(pmcids)}")
    print(f"  Abstract only (PubMed):    {len(pmids_without_pmc)}")

    # ── Step 2: fetch + index PMC full-text articles ─────────────────────────
    pmc_papers: list[dict] = []
    if pmcids:
        time.sleep(0.34)
        pmc_xml = fetch_pmc_fulltext(pmcids)
        pmc_papers = parse_pmc_xml(pmc_xml)

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
        pubmed_xml = fetch_pubmed_details(pmids_without_pmc)
        abstract_papers = parse_pubmed_xml(pubmed_xml)

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
    print(f"\nIngestion complete — {total} papers indexed")
    print(engine.get_corpus_stats())

    return engine


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    engine = ingest_pubmed(
        query="alzheimer's disease biomarkers",
        max_results=5,
    )

    result = engine.query("What biomarkers predict alzheimer's disease?")
    print("\n--- ANSWER ---\n")
    print(result.answer)

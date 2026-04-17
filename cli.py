"""
BioRAG CLI — Interactive decision-support terminal interface.
Run: python cli.py
Or:  python cli.py --query "What biomarkers predict cardiovascular risk in diabetes?"
"""

import sys
import os
import time
import textwrap
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.rag_engine import BioRAGEngine, DecisionOutput
from data.sample_corpus import SAMPLE_DOCUMENTS
from ingestion_pubmed import ingest_pubmed

# ─── ANSI Colors ─────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
BLUE   = "\033[34m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
PURPLE = "\033[35m"
WHITE  = "\033[37m"

def c(text, color): return f"{color}{text}{RESET}"
def header(text): return f"\n{BOLD}{CYAN}{'━'*60}{RESET}\n{BOLD}{WHITE}  {text}{RESET}\n{BOLD}{CYAN}{'━'*60}{RESET}"
def section(text): return f"\n{BOLD}{BLUE}▶ {text}{RESET}"
def dim(text): return f"{DIM}{text}{RESET}"


# ─── Display Helpers ─────────────────────────────────────────────────────────

SUPPORT_COLORS = {
    "direct": GREEN,
    "indirect": YELLOW,
    "contradictory": RED,
}

SUPPORT_ICONS = {
    "direct": "●",
    "indirect": "◐",
    "contradictory": "○",
}

CONFIDENCE_COLORS = {
    "High": GREEN,
    "Moderate": YELLOW,
    "Low": YELLOW,
    "Insufficient": RED,
}

CONFIDENCE_BARS = {
    "High": "████████████ High",
    "Moderate": "████████░░░░ Moderate",
    "Low": "████░░░░░░░░ Low",
    "Insufficient": "██░░░░░░░░░░ Insufficient",
}


def render_output(output: DecisionOutput):
    """Render a DecisionOutput to terminal."""

    # ── Header
    print(header("BioRAG Decision-Support Output"))

    # ── Query
    print(section("Query"))
    print(f"  {BOLD}{output.query}{RESET}")

    # ── Confidence
    print(section("Confidence Assessment"))
    bar = CONFIDENCE_BARS.get(output.confidence_label, "░░░░░░░░░░░░")
    color = CONFIDENCE_COLORS.get(output.confidence_label, WHITE)
    print(f"  {c(bar, color)}  ({output.confidence:.0%})")
    print(f"  {dim(f'Sources used: {output.sources_used}  |  Chunks searched: {output.total_chunks_searched}')}")

    # ── Reasoning Chain
    print(section("Reasoning Chain"))
    for step in output.reasoning_chain:
        conf_bar = "●" * int(step.confidence * 5) + "○" * (5 - int(step.confidence * 5))
        print(f"  {BOLD}{step.step_number}.{RESET} {CYAN}{step.label}{RESET}  {dim(conf_bar)}")
        wrapped = textwrap.fill(step.content, width=70, initial_indent="     ", subsequent_indent="     ")
        print(f"{dim(wrapped)}")

    # ── Answer
    print(section("Synthesized Answer"))
    for part in output.answer.split('\n\n'):
        if part.strip():
            wrapped = textwrap.fill(part.strip(), width=72,
                                    initial_indent="  ", subsequent_indent="  ")
            print(wrapped)
            print()

    # ── Evidence
    print(section(f"Evidence ({len(output.evidence)} nodes)"))
    for e in output.evidence:
        color = SUPPORT_COLORS.get(e.support_type, WHITE)
        icon = SUPPORT_ICONS.get(e.support_type, "○")
        label = e.support_type.upper()
        print(f"\n  {c(icon, color)} {BOLD}{e.doc_title}{RESET}  {c(f'[{label}]', color)}")
        print(f"    {dim(f'Section: {e.section}  |  Relevance: {e.relevance_score:.0%}')}")
        if e.key_terms:
            print(f"    {dim('Key terms:')} {', '.join(e.key_terms[:5])}")
        excerpt = textwrap.fill(e.excerpt[:300], width=68,
                                initial_indent="    ", subsequent_indent="    ")
        print(f"{dim(excerpt)}")

    # ── Knowledge Gaps
    if output.knowledge_gaps:
        print(section("Knowledge Gaps"))
        for gap in output.knowledge_gaps:
            print(f"  {c('⚠', YELLOW)} {gap}")

    # ── Follow-up Questions
    if output.follow_up_questions:
        print(section("Suggested Follow-up Questions"))
        for i, q in enumerate(output.follow_up_questions, 1):
            print(f"  {dim(str(i)+'.')} {q}")

    print(f"\n{DIM}{'─'*60}{RESET}\n")


# ─── Engine Setup ─────────────────────────────────────────────────────────────

def build_engine() -> BioRAGEngine:
    print(f"{DIM}Initializing BioRAG engine...{RESET}", end=" ", flush=True)
    engine = BioRAGEngine()
    total_chunks = 0
    for doc in SAMPLE_DOCUMENTS:
        n = engine.add_document(doc["id"], doc["title"], doc["text"], doc.get("metadata"))
        total_chunks += n
    stats = engine.get_corpus_stats()
    print(f"{GREEN}ready{RESET}")
    print(f"{DIM}  Corpus: {stats['documents']} documents, {stats['chunks']} chunks, "
          f"{stats['unique_terms']:,} unique terms{RESET}\n")
    return engine


# ─── Main ─────────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "What biomarkers predict cardiovascular risk in patients with type 2 diabetes?",
    "How does PD-L1 expression relate to immunotherapy response in lung cancer?",
    "What are the treatment options for carbapenem-resistant Klebsiella pneumoniae?",
    "Which plasma biomarkers are most accurate for early Alzheimer's disease detection?",
    "Compare ceftazidime-avibactam versus colistin for CRE infections",
]


def interactive_loop(engine: BioRAGEngine):
    """Run interactive REPL."""
    print(header("BioRAG Decision-Support System — Interactive Mode"))
    print(f"  {dim('Type a clinical/scientific question, or:')}")
    print(f"  {dim('  ingest <query> [N]  — fetch N PubMed/PMC papers (default 10)')}")
    print(f"  {dim('  demo            — run demonstration queries')}")
    print(f"  {dim('  stats           — show corpus statistics')}")
    print(f"  {dim('  quit            — exit')}")

    while True:
        try:
            query = input(f"\n{BOLD}{CYAN}Query>{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        if query.lower().startswith("ingest "):
            parts = query[7:].strip().rsplit(maxsplit=1)
            if not parts:
                print(f"  {c('Usage:', YELLOW)} ingest <search query> [N]")
                continue
            # Allow an optional trailing integer as the paper count
            if len(parts) == 2 and parts[1].isdigit():
                pubmed_query, max_results = parts[0], int(parts[1])
            else:
                pubmed_query, max_results = " ".join(parts), 10
            print(f"{DIM}Fetching {max_results} papers from PubMed/PMC: {pubmed_query}{RESET}")
            try:
                ingest_pubmed(pubmed_query, max_results=max_results, engine=engine)
                stats = engine.get_corpus_stats()
                print(
                    f"{GREEN}Corpus updated:{RESET} "
                    f"{stats['documents']} docs · "
                    f"{stats['full_text_documents']} full-text · "
                    f"{stats['abstract_only_documents']} abstract-only · "
                    f"{stats['chunks']} chunks"
                )
            except Exception as exc:
                print(f"{RED}Ingestion failed:{RESET} {exc}")
            continue

        if query.lower() == "stats":
            stats = engine.get_corpus_stats()
            print(json_pretty(stats))
            continue

        if query.lower() == "demo":
            for dq in DEMO_QUERIES:
                print(f"\n{dim('Running:')} {dq}")
                t0 = time.perf_counter()
                out = engine.query(dq)
                elapsed = (time.perf_counter() - t0) * 1000
                render_output(out)
                print(f"{dim(f'Latency: {elapsed:.0f}ms')}")
            continue

        t0 = time.perf_counter()
        out = engine.query(query)
        elapsed = (time.perf_counter() - t0) * 1000
        render_output(out)
        print(f"{dim(f'Latency: {elapsed:.0f}ms')}")


def json_pretty(obj):
    import json
    return json.dumps(obj, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="BioRAG CLI")
    parser.add_argument("--query", "-q", type=str, help="Run a single query and exit")
    parser.add_argument("--demo", action="store_true", help="Run demonstration queries")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics")
    parser.add_argument(
        "--ingest", metavar="PUBMED_QUERY",
        help="Fetch PubMed/PMC papers for PUBMED_QUERY, add to corpus, then enter interactive mode",
    )
    parser.add_argument(
        "--ingest-max", metavar="N", type=int, default=10,
        help="Number of papers to fetch when using --ingest (default: 10)",
    )
    args = parser.parse_args()

    engine = build_engine()

    if args.ingest:
        print(f"{DIM}Fetching {args.ingest_max} papers from PubMed/PMC: {args.ingest}{RESET}")
        try:
            ingest_pubmed(args.ingest, max_results=args.ingest_max, engine=engine)
            stats = engine.get_corpus_stats()
            print(
                f"{GREEN}Corpus updated:{RESET} "
                f"{stats['documents']} docs · "
                f"{stats['full_text_documents']} full-text · "
                f"{stats['abstract_only_documents']} abstract-only · "
                f"{stats['chunks']} chunks\n"
            )
        except Exception as exc:
            print(f"{RED}Ingestion failed:{RESET} {exc}")

    if args.stats:
        import json
        print(json.dumps(engine.get_corpus_stats(), indent=2))
        return

    if args.demo:
        for dq in DEMO_QUERIES:
            t0 = time.perf_counter()
            out = engine.query(dq)
            elapsed = (time.perf_counter() - t0) * 1000
            render_output(out)
            print(f"{dim(f'Latency: {elapsed:.0f}ms')}")
        return

    if args.query:
        t0 = time.perf_counter()
        out = engine.query(args.query)
        elapsed = (time.perf_counter() - t0) * 1000
        render_output(out)
        print(f"{dim(f'Latency: {elapsed:.0f}ms')}")
        return

    interactive_loop(engine)


if __name__ == "__main__":
    import json
    main()

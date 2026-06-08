"""
RAGAS-backed Answer Quality Eval
══════════════════════════════════
Implements the same five-dimension rubric as evals/answer_eval.py, but uses
RAGAS metrics as the judge instead of a hand-crafted tool_use prompt.

Metric mapping
──────────────
  Semantic Coverage (0-2)     → LabelledRubricsScore  (RAGAS returns 1-3, mapped to 0-2)
  Entity Coverage   (0-2)     → LabelledRubricsScore  (RAGAS returns 1-3, mapped to 0-2)
  Directional Agreement (0-1) → AspectCritique (binary)
  Quantitative Detail   (0-1) → AspectCritique (binary)
  Contextual Accuracy   (0-1) → AspectCritique (binary)

The LLM judge is Claude (via LangChain's ChatAnthropic wrapper) — the same
model used in answer_eval.py — so the scores are directly comparable.

Usage
─────
  python evals/ragas_answer_eval.py
  python evals/ragas_answer_eval.py --alzheimer-only
  python evals/ragas_answer_eval.py --compare          # side-by-side vs answer_eval.py
"""

import sys
import os
import asyncio
import argparse
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LabelledRubricsScore, AspectCritique
from ragas.metrics.base import EvaluationMode
from ragas import evaluate

from core.rag_engine import BioRAGEngine, DecisionOutput, EvidenceNode
from data.sample_corpus import SAMPLE_DOCUMENTS
from evals.ground_truth import EVAL_QUERIES, ALZHEIMER_QUERIES, RetrievalQuery
from evals.answer_ground_truth import ANSWER_CLAIMS, ALZHEIMER_CLAIMS, AnswerClaim


# ─── RAGAS Metric Definitions ─────────────────────────────────────────────────

def build_metrics(llm: LangchainLLMWrapper) -> dict:
    """Build all five rubric metrics wired to the given LLM judge."""

    semantic_coverage = LabelledRubricsScore(
        name="semantic_coverage",
        llm=llm,
        rubrics={
            "score1_description": (
                "The answer only covers the broad topic area without addressing "
                "the specific biological phenomenon described in the reference claim."
            ),
            "score2_description": (
                "The answer partially engages with the specific biological phenomenon "
                "described in the reference claim, but is incomplete or addresses it "
                "at a different granularity."
            ),
            "score3_description": (
                "The answer directly and substantively asserts the same biological "
                "conclusion as the reference claim — same phenomenon, same specificity."
            ),
        },
    )

    entity_coverage = LabelledRubricsScore(
        name="entity_coverage",
        llm=llm,
        rubrics={
            "score1_description": (
                "None of the specific biomedical entities named in the reference claim "
                "(genes, proteins, drugs, biomarkers, etc.) appear in the answer."
            ),
            "score2_description": (
                "Some of the expected entities are named, or all are named but in an "
                "irrelevant or incorrect biological context."
            ),
            "score3_description": (
                "All key entities from the reference claim are explicitly named and "
                "used in the correct biological context in the answer."
            ),
        },
    )

    directional_agreement = AspectCritique(
        name="directional_agreement",
        llm=llm,
        definition=(
            "Does the answer explicitly state the direction of effect described in the "
            "ground truth reference claim? The direction must be stated explicitly "
            "(e.g. 'reduces', 'is elevated', 'has no significant effect') — not merely "
            "implied or absent. Score 1 if direction is explicit and matches, 0 otherwise."
        ),
    )

    quantitative_detail = AspectCritique(
        name="quantitative_detail",
        llm=llm,
        definition=(
            "Does the answer include at least one quantitative value (effect size, "
            "p-value, AUC, WMD, proportion, count, etc.) that is consistent with the "
            "ground truth reference claim? Exact values are not required — general "
            "magnitudes consistent with the claim qualify. Score 1 if yes, 0 if no "
            "quantitative information is present or the magnitudes are inconsistent."
        ),
    )

    contextual_accuracy = AspectCritique(
        name="contextual_accuracy",
        llm=llm,
        definition=(
            "Does the answer explicitly name the specific context from the ground truth "
            "reference claim — such as the patient subgroup, tissue type, timepoint, "
            "condition, or treatment setting? Generic phrases like 'in patients' or "
            "'in the disease context' do not qualify. Score 1 if the specific context "
            "is explicitly named, 0 otherwise."
        ),
    )

    return {
        "semantic_coverage": semantic_coverage,
        "entity_coverage": entity_coverage,
        "directional_agreement": directional_agreement,
        "quantitative_detail": quantitative_detail,
        "contextual_accuracy": contextual_accuracy,
    }


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RagasRubricScores:
    semantic_coverage: float      # 0-2 (mapped from RAGAS 1-3)
    entity_coverage: float        # 0-2 (mapped from RAGAS 1-3)
    directional_agreement: float  # 0 or 1
    quantitative_detail: float    # 0 or 1
    contextual_accuracy: float    # 0 or 1

    @property
    def total(self) -> float:
        return (
            self.semantic_coverage
            + self.entity_coverage
            + self.directional_agreement
            + self.quantitative_detail
            + self.contextual_accuracy
        )


@dataclass
class RagasQueryResult:
    query_id: str
    query: str
    rubric: RagasRubricScores


@dataclass
class RagasEvalReport:
    n_queries: int
    mean_total: float
    mean_by_dimension: dict
    per_query: list


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def _evidence_to_contexts(evidence: list[EvidenceNode]) -> list[str]:
    """Format evidence nodes as plain-text context strings for RAGAS."""
    return [
        f"[{e.doc_title} / {e.section}] {e.excerpt}"
        for e in evidence
    ]


def _enrich_ground_truth(claim: AnswerClaim) -> str:
    """Combine reference claim with expected entities/direction/context."""
    return (
        f"{claim.reference_claim}\n\n"
        f"Expected entities: {', '.join(claim.expected_entities)}.\n"
        f"Expected direction: {claim.expected_direction}.\n"
        f"Expected context: {claim.expected_context}."
    )


def build_ragas_dataset(
    pairs: list[tuple[RetrievalQuery, AnswerClaim]],
    engine: BioRAGEngine,
) -> tuple[Dataset, list[str]]:
    """
    Run engine.query() for each pair and format results as a RAGAS Dataset.

    Returns (dataset, query_ids) so results can be matched back to queries.
    """
    rows = []
    query_ids = []

    for i, (rq, claim) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {rq.query_id} — {rq.query[:60]}…", flush=True)
        output: DecisionOutput = engine.query(rq.query)
        rows.append({
            "question": rq.query,
            "answer": output.answer,
            "contexts": _evidence_to_contexts(output.evidence) or ["No evidence retrieved."],
            "ground_truth": _enrich_ground_truth(claim),
        })
        query_ids.append(rq.query_id)

    return Dataset.from_list(rows), query_ids


# ─── Evaluator ────────────────────────────────────────────────────────────────

class RagasAnswerEvaluator:
    """
    Runs the five-dimension rubric via RAGAS metrics.

    LabelledRubricsScore uses EvaluationMode.qcg (question + contexts + ground_truth)
    but actually reads all four row fields including 'answer'. AspectCritique uses
    EvaluationMode.qac (question + answer + contexts).
    """

    def __init__(self, engine: BioRAGEngine, model: str = "claude-sonnet-4-6") -> None:
        self.engine = engine
        langchain_llm = ChatAnthropic(model=model, temperature=0)
        self.llm = LangchainLLMWrapper(langchain_llm)
        self.metrics = build_metrics(self.llm)

    def evaluate(
        self,
        queries: list[RetrievalQuery],
        claims: list[AnswerClaim],
    ) -> RagasEvalReport:
        claim_by_id = {c.query_id: c for c in claims}
        pairs = [(q, claim_by_id[q.query_id]) for q in queries if q.query_id in claim_by_id]

        print("Building dataset (running engine queries)…")
        dataset, query_ids = build_ragas_dataset(pairs, self.engine)

        print("Running RAGAS evaluation…")
        result = evaluate(
            dataset,
            metrics=list(self.metrics.values()),
        )

        # Build per-query results — RAGAS returns a dict of metric -> list of scores
        scores_df = result.to_pandas()
        per_query = []
        for i, qid in enumerate(query_ids):
            raw_sem = scores_df["semantic_coverage"].iloc[i]
            raw_ent = scores_df["entity_coverage"].iloc[i]

            # LabelledRubricsScore returns 1-3; map to 0-2
            sem = max(0.0, float(raw_sem) - 1.0)
            ent = max(0.0, float(raw_ent) - 1.0)

            rubric = RagasRubricScores(
                semantic_coverage=sem,
                entity_coverage=ent,
                directional_agreement=float(scores_df["directional_agreement"].iloc[i]),
                quantitative_detail=float(scores_df["quantitative_detail"].iloc[i]),
                contextual_accuracy=float(scores_df["contextual_accuracy"].iloc[i]),
            )
            per_query.append(RagasQueryResult(
                query_id=qid,
                query=pairs[i][0].query,
                rubric=rubric,
            ))

        n = len(per_query)
        mean_total = sum(r.rubric.total for r in per_query) / n
        dims = ["semantic_coverage", "entity_coverage", "directional_agreement",
                "quantitative_detail", "contextual_accuracy"]
        mean_by_dim = {
            d: sum(getattr(r.rubric, d) for r in per_query) / n
            for d in dims
        }

        return RagasEvalReport(
            n_queries=n,
            mean_total=mean_total,
            mean_by_dimension=mean_by_dim,
            per_query=per_query,
        )


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_ragas_report(report: RagasEvalReport) -> None:
    W = 72
    print(f"\n{'━' * W}")
    print(f"  BioRAG Answer Quality Eval — RAGAS judge  ({report.n_queries} queries)")
    print(f"{'━' * W}")

    dim_labels = {
        "semantic_coverage":     ("Semantic Coverage",     2),
        "entity_coverage":       ("Entity Coverage",       2),
        "directional_agreement": ("Directional Agreement", 1),
        "quantitative_detail":   ("Quantitative Detail",   1),
        "contextual_accuracy":   ("Contextual Accuracy",   1),
    }
    print(f"\n  {'Dimension':<26} {'Mean':>6}  {'Max':>4}")
    print(f"  {'─' * 44}")
    for key, (label, max_v) in dim_labels.items():
        mean = report.mean_by_dimension[key]
        print(f"  {label:<26} {mean:>6.2f}  /{max_v}")
    print(f"  {'─' * 44}")
    print(f"  {'Total Score':<26} {report.mean_total:>6.2f}  /7")

    print(f"\n  {'ID':<5} {'Sem':>4} {'Ent':>4} {'Dir':>4} {'Qty':>4} {'Ctx':>4} {'Tot':>5}")
    print(f"  {'─' * 40}")
    for r in report.per_query:
        rb = r.rubric
        print(
            f"  {r.query_id:<5} {rb.semantic_coverage:>4.1f} {rb.entity_coverage:>4.1f} "
            f"{rb.directional_agreement:>4.1f} {rb.quantitative_detail:>4.1f} "
            f"{rb.contextual_accuracy:>4.1f} {rb.total:>5.1f}"
        )
    print(f"\n{'━' * W}\n")


def print_comparison(ragas_report: RagasEvalReport) -> None:
    """Side-by-side: our custom judge vs RAGAS judge (hardcoded rule-based results)."""
    # Rule-based results from the previous run
    custom_rule = {
        "Q01": (1, 0, 0, 0, 1, 2),
        "Q02": (1, 1, 0, 0, 1, 3),
        "Q03": (1, 0, 0, 0, 1, 2),
        "Q07": (1, 1, 0, 0, 1, 3),
        "Q08": (2, 2, 1, 1, 1, 7),
        "Q09": (2, 2, 1, 1, 1, 7),
        "Q11": (1, 2, 1, 0, 1, 5),
        "Q12": (1, 1, 0, 0, 1, 3),
        "Q13": (1, 2, 0, 0, 1, 4),
        "Q14": (1, 1, 1, 0, 1, 4),
    }
    custom_llm = {
        "Q01": (1, 1, 0, 0, 1, 3),
        "Q02": (1, 1, 0, 0, 1, 3),
        "Q03": (1, 1, 1, 0, 1, 4),
        "Q07": (2, 2, 1, 0, 1, 6),
        "Q08": (2, 2, 1, 1, 1, 7),
        "Q09": (2, 2, 1, 1, 1, 7),
        "Q11": (1, 2, 1, 0, 1, 5),
        "Q12": (1, 1, 1, 0, 1, 4),
        "Q13": (2, 2, 1, 0, 1, 6),
        "Q14": (2, 2, 1, 1, 1, 7),
    }

    W = 88
    print(f"\n{'━' * W}")
    print("  Three-Way Comparison: Custom Judge (rule-based synth) | Custom Judge (LLM synth) | RAGAS Judge (rule-based synth)")
    print(f"{'━' * W}")
    print(f"\n  {'ID':<5} {'Custom/Rule':>12} {'Custom/LLM':>11} {'RAGAS/Rule':>11}  {'Δ RAGAS vs Custom/Rule':>22}")
    print(f"  {'─' * 70}")

    ragas_by_id = {r.query_id: r.rubric.total for r in ragas_report.per_query}
    for qid in sorted(ragas_by_id.keys()):
        r_tot = ragas_by_id[qid]
        c_rule = custom_rule.get(qid, (0,)*6)[-1]
        c_llm  = custom_llm.get(qid, (0,)*6)[-1]
        delta = r_tot - c_rule
        print(f"  {qid:<5} {c_rule:>12} {c_llm:>11} {r_tot:>11.1f}  {delta:>+22.1f}")

    mean_ragas = sum(ragas_by_id.values()) / len(ragas_by_id)
    mean_rule  = sum(v[-1] for v in custom_rule.values()) / len(custom_rule)
    mean_llm   = sum(v[-1] for v in custom_llm.values()) / len(custom_llm)
    print(f"  {'─' * 70}")
    print(f"  {'Mean':<5} {mean_rule:>12.2f} {mean_llm:>11.2f} {mean_ragas:>11.2f}  {mean_ragas - mean_rule:>+22.2f}")
    print(f"\n  Positive Δ = RAGAS judge is more generous than our custom judge.")
    print(f"  Negative Δ = RAGAS judge is stricter.\n")
    print(f"{'━' * W}\n")


# ─── Engine / CLI ─────────────────────────────────────────────────────────────

def build_engine() -> BioRAGEngine:
    engine = BioRAGEngine()
    for doc in SAMPLE_DOCUMENTS:
        engine.add_document(
            doc["id"], doc["title"], doc["text"], doc.get("metadata", {})
        )
    return engine


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BioRAG answer quality eval — RAGAS judge"
    )
    parser.add_argument("--alzheimer-only", action="store_true")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print three-way comparison: custom judge (rule/LLM) vs RAGAS judge",
    )
    args = parser.parse_args()

    queries = ALZHEIMER_QUERIES if args.alzheimer_only else EVAL_QUERIES
    claims  = ALZHEIMER_CLAIMS  if args.alzheimer_only else ANSWER_CLAIMS

    print("Loading corpus…", end=" ", flush=True)
    engine = build_engine()
    stats = engine.get_corpus_stats()
    print(f"done  ({stats['documents']} docs, {stats['chunks']} chunks)")

    evaluator = RagasAnswerEvaluator(engine)
    report = evaluator.evaluate(queries, claims)

    print_ragas_report(report)

    if args.compare:
        print_comparison(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())

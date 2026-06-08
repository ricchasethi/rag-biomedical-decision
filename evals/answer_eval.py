"""
Answer Quality Eval — LLM-as-Judge
════════════════════════════════════
Evaluates the prose answer produced by BioRAGEngine.query() against a
hand-authored reference claim using Claude as an independent judge.

The judge scores each answer on five rubric dimensions (from ideas.md):

  Semantic Coverage    (0–2)  Does the answer address the right phenomenon?
  Entity Coverage      (0–2)  Are the specific genes/markers/drugs named?
  Directional Agreement (0–1) Does the stated direction of effect match?
  Quantitative Detail   (0–1) Are magnitudes / statistics consistent?
  Contextual Accuracy   (0–1) Is the finding placed in the correct context?

Maximum total: 7 points per query.

This is intentionally separate from the retrieval eval (MRR/NDCG) so we can
show the key insight: good retrieval does not guarantee good answers.  A
system can retrieve the correct document at rank 1 (MRR=1.0) while still
producing an answer that misses the direction of effect or omits key entities.

Usage
─────
  # Default: score all 10 reference claims
  python evals/answer_eval.py

  # Alzheimer's subset only
  python evals/answer_eval.py --alzheimer-only

  # Show full answers and judge rationale per query
  python evals/answer_eval.py --verbose

  # Use ClaudeSynthesizer for answers (requires ANTHROPIC_API_KEY)
  python evals/answer_eval.py --llm

  # Show combined retrieval + answer quality table
  python evals/answer_eval.py --with-retrieval
"""

import sys
import os
import json
import argparse
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_engine import BioRAGEngine, DecisionOutput
from data.sample_corpus import SAMPLE_DOCUMENTS
from evals.ground_truth import EVAL_QUERIES, ALZHEIMER_QUERIES, RetrievalQuery
from evals.answer_ground_truth import (
    ANSWER_CLAIMS,
    ALZHEIMER_CLAIMS,
    AnswerClaim,
)


# ─── Judge tool schema ────────────────────────────────────────────────────────
#
# Using tool_use forces Claude to return integer scores in the declared enum
# values — more reliable than asking for raw JSON in the message text.

_JUDGE_TOOL: dict = {
    "name": "score_answer",
    "description": (
        "Score the AI-generated answer on five biomedical evaluation dimensions "
        "against a reference claim. Fill every field; do not omit any."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "semantic_coverage": {
                "type": "integer",
                "enum": [0, 1, 2],
                "description": (
                    "0 = only the broad topic covered, not the specific phenomenon. "
                    "1 = engages the specific phenomenon but incompletely. "
                    "2 = same biological conclusion as the reference claim."
                ),
            },
            "entity_coverage": {
                "type": "integer",
                "enum": [0, 1, 2],
                "description": (
                    "0 = none of the claim's specific entities are named. "
                    "1 = some entities named, or all named but in wrong context. "
                    "2 = all key entities addressed in the relevant context."
                ),
            },
            "directional_agreement": {
                "type": "integer",
                "enum": [0, 1],
                "description": (
                    "0 = direction absent, unclear, or contradicted. "
                    "1 = stated direction of effect matches the reference claim."
                ),
            },
            "quantitative_detail": {
                "type": "integer",
                "enum": [0, 1],
                "description": (
                    "0 = no quantitative information given, or magnitudes inconsistent. "
                    "1 = general magnitudes or statistics consistent with the claim "
                    "(exact numbers need not match precisely)."
                ),
            },
            "contextual_accuracy": {
                "type": "integer",
                "enum": [0, 1],
                "description": (
                    "0 = wrong context, no context, or only generic language. "
                    "1 = answer explicitly names the claim-specific context "
                    "(timepoint, treatment arm, tissue, subgroup, etc.)."
                ),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "1–2 sentences explaining the scores, noting what the answer "
                    "got right and what it missed."
                ),
            },
        },
        "required": [
            "semantic_coverage",
            "entity_coverage",
            "directional_agreement",
            "quantitative_detail",
            "contextual_accuracy",
            "rationale",
        ],
    },
}

_JUDGE_SYSTEM_PROMPT = """\
You are an expert biomedical evaluator assessing the quality of an \
AI-generated answer against a reference claim.

Your task is to call the score_answer tool with integer scores for each of \
the five rubric dimensions. Base your scores strictly on what the AI answer \
explicitly states — do not infer or credit implied knowledge.

Scoring rules:
- Semantic coverage 2: the answer asserts the same specific biological \
conclusion as the reference claim, not just the same broad topic.
- Entity coverage 2: every expected entity is named AND used in the correct \
context. Synonyms count only if they are true functional equivalents \
(e.g. "phospho-tau" for "p-tau"), not just same family.
- Directional agreement 1: the stated direction must be explicit (e.g. \
"reduces", "elevated", "no significant change") and match the reference. \
Absence of direction scores 0.
- Quantitative detail 1: at least one magnitude, effect size, AUC, or \
statistic consistent with the reference is present. Exact values are not \
required.
- Contextual accuracy 1: the answer names the specific setting (tissue, \
condition, subgroup, timepoint) from the reference claim. Generic phrases \
like "in patients" or "in disease" do not qualify.

Call score_answer once with all five scores plus a brief rationale.\
"""


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RubricScores:
    """Per-dimension scores returned by the judge for one answer."""
    semantic_coverage: int      # 0–2
    entity_coverage: int        # 0–2
    directional_agreement: int  # 0–1
    quantitative_detail: int    # 0–1
    contextual_accuracy: int    # 0–1
    rationale: str

    @property
    def total(self) -> int:
        return (
            self.semantic_coverage
            + self.entity_coverage
            + self.directional_agreement
            + self.quantitative_detail
            + self.contextual_accuracy
        )

    @property
    def max_score(self) -> int:
        return 7


@dataclass
class AnswerQueryResult:
    """Full eval result for one query."""
    query_id: str
    query: str
    answer: str
    engine_confidence: float
    rubric: RubricScores


@dataclass
class AnswerEvalReport:
    """Aggregated report returned by AnswerEvaluator.evaluate()."""
    n_queries: int
    synthesizer: str              # "rule-based" or "claude-sonnet-4-6"
    mean_total: float             # mean of rubric.total across queries
    mean_by_dimension: dict[str, float]
    per_query: list[AnswerQueryResult]


# ─── Evaluator ────────────────────────────────────────────────────────────────

class AnswerEvaluator:
    """
    Runs BioRAGEngine.query() for each eval query and judges the answer
    using Claude as an independent rubric scorer.

    The judge is isolated from the synthesizer — it only sees the answer text
    plus the reference claim, not the retrieval context.  This is intentional:
    we want to score what the user would read, not what the pipeline saw.
    """

    def __init__(self, engine: BioRAGEngine, model: str = "claude-sonnet-4-6") -> None:
        self.engine = engine
        self.model = model
        self._client = None

    @property
    def _anthropic_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "anthropic package is required for AnswerEvaluator. "
                    "Install it with: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    # ── Judging ───────────────────────────────────────────────────────────────

    def _build_judge_user_message(self, answer: str, claim: AnswerClaim) -> str:
        """Format the reference claim and AI answer for the judge."""
        return (
            f"## Reference claim\n{claim.reference_claim}\n\n"
            f"## Expected entities (must ALL be named for full entity-coverage credit)\n"
            + ", ".join(claim.expected_entities) + "\n\n"
            f"## Expected direction of effect\n{claim.expected_direction}\n\n"
            f"## Expected context (must be named explicitly)\n{claim.expected_context}\n\n"
            f"## AI-generated answer to evaluate\n{answer}\n\n"
            "Score the answer on all five dimensions using the score_answer tool."
        )

    def _judge_answer(self, answer: str, claim: AnswerClaim) -> RubricScores:
        """Call Claude to score the answer; return RubricScores."""
        user_msg = self._build_judge_user_message(answer, claim)

        response = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0,
            system=_JUDGE_SYSTEM_PROMPT,
            tools=[_JUDGE_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_msg}],
        )

        # Extract the tool_use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "score_answer":
                inp = block.input
                return RubricScores(
                    semantic_coverage=int(inp["semantic_coverage"]),
                    entity_coverage=int(inp["entity_coverage"]),
                    directional_agreement=int(inp["directional_agreement"]),
                    quantitative_detail=int(inp["quantitative_detail"]),
                    contextual_accuracy=int(inp["contextual_accuracy"]),
                    rationale=inp.get("rationale", ""),
                )

        # Fallback if tool_use block absent (should not happen with tool_choice=any)
        raise RuntimeError(
            f"Judge returned no tool_use block for query {claim.query_id}. "
            f"Response: {response}"
        )

    # ── Per-query eval ────────────────────────────────────────────────────────

    def evaluate_query(
        self, rq: RetrievalQuery, claim: AnswerClaim
    ) -> AnswerQueryResult:
        """Run the full engine pipeline, then judge the answer."""
        output: DecisionOutput = self.engine.query(rq.query)
        rubric = self._judge_answer(output.answer, claim)
        return AnswerQueryResult(
            query_id=rq.query_id,
            query=rq.query,
            answer=output.answer,
            engine_confidence=output.confidence,
            rubric=rubric,
        )

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(
        self, results: list[AnswerQueryResult]
    ) -> tuple[float, dict[str, float]]:
        """Mean total score and per-dimension means."""
        n = len(results)
        mean_total = sum(r.rubric.total for r in results) / n
        dims = ["semantic_coverage", "entity_coverage", "directional_agreement",
                "quantitative_detail", "contextual_accuracy"]
        mean_by_dim = {
            d: sum(getattr(r.rubric, d) for r in results) / n
            for d in dims
        }
        return mean_total, mean_by_dim

    # ── Main entry point ──────────────────────────────────────────────────────

    def evaluate(
        self,
        queries: list[RetrievalQuery],
        claims: list[AnswerClaim],
    ) -> AnswerEvalReport:
        """Evaluate all queries that have a matching claim."""
        claim_by_id = {c.query_id: c for c in claims}
        pairs = [(q, claim_by_id[q.query_id]) for q in queries if q.query_id in claim_by_id]

        if not pairs:
            raise ValueError("No queries matched any claim query_ids.")

        synth_name = getattr(self.engine.synthesizer, "model", "rule-based")
        results: list[AnswerQueryResult] = []
        for i, (q, c) in enumerate(pairs, 1):
            print(f"  [{i}/{len(pairs)}] {q.query_id} — {q.query[:60]}…", flush=True)
            results.append(self.evaluate_query(q, c))

        mean_total, mean_by_dim = self._aggregate(results)
        return AnswerEvalReport(
            n_queries=len(results),
            synthesizer=synth_name,
            mean_total=mean_total,
            mean_by_dimension=mean_by_dim,
            per_query=results,
        )


# ─── Reporting ────────────────────────────────────────────────────────────────

def _bar(score: int, max_score: int) -> str:
    """Compact filled/empty bar for terminal sparkline."""
    filled = round(score / max_score * 5)
    return "█" * filled + "░" * (5 - filled)


def print_report(report: AnswerEvalReport, verbose: bool = False) -> None:
    """Print the answer quality report to stdout."""
    W = 72
    print(f"\n{'━' * W}")
    print(f"  BioRAG Answer Quality Eval  ({report.n_queries} queries)")
    print(f"  Synthesizer: {report.synthesizer}")
    print(f"{'━' * W}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {'Dimension':<26} {'Mean':>6}  {'Max':>4}  Bar")
    print(f"  {'─' * 50}")
    dim_labels = {
        "semantic_coverage":     ("Semantic Coverage",     2),
        "entity_coverage":       ("Entity Coverage",       2),
        "directional_agreement": ("Directional Agreement", 1),
        "quantitative_detail":   ("Quantitative Detail",   1),
        "contextual_accuracy":   ("Contextual Accuracy",   1),
    }
    for key, (label, max_v) in dim_labels.items():
        mean = report.mean_by_dimension[key]
        bar = _bar(round(mean * 5 / max_v), 5)
        print(f"  {label:<26} {mean:>6.2f}  /{max_v:<3}  {bar}")

    print(f"  {'─' * 50}")
    print(f"  {'Total Score':<26} {report.mean_total:>6.2f}  /7")
    print(f"\n  Note: total = sum of all five dimensions (max 7 per query)")

    # ── Per-query table ───────────────────────────────────────────────────────
    print(f"\n  {'ID':<5} {'Sem':>4} {'Ent':>4} {'Dir':>4} {'Qty':>4} {'Ctx':>4} {'Tot':>5}  Conf")
    print(f"  {'─' * 48}")
    for r in report.per_query:
        rb = r.rubric
        print(
            f"  {r.query_id:<5} {rb.semantic_coverage:>4} {rb.entity_coverage:>4} "
            f"{rb.directional_agreement:>4} {rb.quantitative_detail:>4} "
            f"{rb.contextual_accuracy:>4} {rb.total:>5}  {r.engine_confidence:.2f}"
        )

    # ── Verbose: full answer + rationale ─────────────────────────────────────
    if verbose:
        print(f"\n  {'Per-Query Detail'}")
        print(f"  {'─' * W}")
        for r in report.per_query:
            print(f"\n  [{r.query_id}]  {r.query}")
            print(f"  Answer (first 300 chars):")
            print(f"    {r.answer[:300].replace(chr(10), ' ')}")
            print(f"  Rubric scores: "
                  f"Sem={r.rubric.semantic_coverage}/2  "
                  f"Ent={r.rubric.entity_coverage}/2  "
                  f"Dir={r.rubric.directional_agreement}/1  "
                  f"Qty={r.rubric.quantitative_detail}/1  "
                  f"Ctx={r.rubric.contextual_accuracy}/1  "
                  f"Total={r.rubric.total}/7")
            print(f"  Rationale: {r.rubric.rationale}")

    print(f"\n{'━' * W}\n")


def print_combined_report(
    answer_report: AnswerEvalReport,
    retrieval_results: dict,   # query_id -> (mrr, ndcg3) from retrieval eval
) -> None:
    """Side-by-side table: retrieval metrics vs answer quality."""
    W = 80
    print(f"\n{'━' * W}")
    print("  Combined: Retrieval Quality vs Answer Quality")
    print(f"{'━' * W}")
    print(
        f"\n  {'ID':<5} {'MRR':>6} {'NDCG@3':>7} {'Sem':>4} {'Ent':>4} "
        f"{'Dir':>4} {'Qty':>4} {'Ctx':>4} {'Tot/7':>6}"
    )
    print(f"  {'─' * 60}")
    for r in answer_report.per_query:
        mrr, ndcg3 = retrieval_results.get(r.query_id, (0.0, 0.0))
        rb = r.rubric
        print(
            f"  {r.query_id:<5} {mrr:>6.3f} {ndcg3:>7.3f} "
            f"{rb.semantic_coverage:>4} {rb.entity_coverage:>4} "
            f"{rb.directional_agreement:>4} {rb.quantitative_detail:>4} "
            f"{rb.contextual_accuracy:>4} {rb.total:>6}"
        )
    print(f"\n  Interpretation: high MRR with low answer score = retrieval works,")
    print(f"  synthesis fails.  Low MRR with any answer score = retrieval is the bottleneck.")
    print(f"\n{'━' * W}\n")


# ─── Engine builders ─────────────────────────────────────────────────────────

def build_engine(use_llm: bool = False) -> BioRAGEngine:
    """Load sample corpus into a BioRAGEngine, optionally with ClaudeSynthesizer."""
    if use_llm:
        from llm_synthesizer import ClaudeSynthesizer
        engine = BioRAGEngine(synthesizer=ClaudeSynthesizer())
    else:
        engine = BioRAGEngine()

    for doc in SAMPLE_DOCUMENTS:
        engine.add_document(
            doc["id"], doc["title"], doc["text"], doc.get("metadata", {})
        )
    return engine


def _build_retrieval_lookup(queries: list[RetrievalQuery]) -> dict[str, tuple[float, float]]:
    """Run the retrieval eval and return {query_id: (mrr, ndcg@3)} for the combined table."""
    from evals.retrieval_eval import RetrievalEvaluator, build_engine as build_ret_engine

    ret_engine = build_ret_engine()
    evaluator = RetrievalEvaluator(ret_engine, ks=[1, 3, 5])
    report = evaluator.evaluate(queries)
    return {
        r.query_id: (r.reranked.rr, r.reranked.ndcg.get(3, 0.0))
        for r in report.per_query
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="BioRAG answer quality eval — LLM-as-judge rubric"
    )
    parser.add_argument(
        "--alzheimer-only",
        action="store_true",
        help="Evaluate only the Alzheimer's disease query subset (Q01–Q07)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full answer text and judge rationale per query",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Use ClaudeSynthesizer to generate answers (default: rule-based). "
            "Requires ANTHROPIC_API_KEY."
        ),
    )
    parser.add_argument(
        "--with-retrieval",
        action="store_true",
        help="Also run retrieval eval and print a combined side-by-side table",
    )
    args = parser.parse_args()

    if args.alzheimer_only:
        queries = ALZHEIMER_QUERIES
        claims = ALZHEIMER_CLAIMS
    else:
        queries = EVAL_QUERIES
        claims = ANSWER_CLAIMS

    print("Loading corpus…", end=" ", flush=True)
    engine = build_engine(use_llm=args.llm)
    stats = engine.get_corpus_stats()
    print(
        f"done  ({stats['documents']} docs, {stats['chunks']} chunks)"
        + (" [LLM synthesizer]" if args.llm else " [rule-based synthesizer]")
    )

    evaluator = AnswerEvaluator(engine)
    print(f"Judging {len([q for q in queries if any(c.query_id == q.query_id for c in claims)])} queries…")
    report = evaluator.evaluate(queries, claims)

    print_report(report, verbose=args.verbose)

    if args.with_retrieval:
        print("Running retrieval eval for combined table…", flush=True)
        ret_lookup = _build_retrieval_lookup(queries)
        print_combined_report(report, ret_lookup)

    return 0


if __name__ == "__main__":
    sys.exit(main())

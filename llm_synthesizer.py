"""
LLM-backed answer synthesizer for BioRAG.

Drop-in replacement for the stdlib AnswerSynthesizer that calls Claude to
generate the final answer text.  All other pipeline stages (BM25 retrieval,
reranker, evidence classifier, gap detector, reasoning chain construction,
confidence scoring) remain identical — only the prose answer is LLM-generated.

Grounding guarantee
-------------------
The system prompt instructs Claude to answer *exclusively* from the numbered
evidence excerpts supplied in the user message.  If the excerpts do not
contain enough information, Claude must say so rather than drawing on training
knowledge.  The prompt is stored in ``self.last_prompt`` after each call so
you can inspect it.

Usage
-----
    from llm_synthesizer import ClaudeSynthesizer
    from core.rag_engine import BioRAGEngine

    engine = BioRAGEngine(synthesizer=ClaudeSynthesizer())
    result = engine.query("What biomarkers predict Alzheimer's disease?")

    # Inspect the last prompt sent to Claude
    print(engine.synthesizer.last_prompt["system"])
    print(engine.synthesizer.last_prompt["user"])
"""

from __future__ import annotations

from core.rag_engine import AnswerSynthesizer, EvidenceNode, ReasoningStep, TextProcessor

# ─── System prompt ────────────────────────────────────────────────────────────
# Kept as a module-level constant so it is easy to audit and version-control.

SYSTEM_PROMPT = """\
You are a biomedical decision-support assistant integrated into a Retrieval-\
Augmented Generation (RAG) system called BioRAG.

## Your task
Answer the researcher's question using ONLY the numbered evidence excerpts \
provided in the user message.  Each excerpt comes from a peer-reviewed paper \
or preprint that has been indexed in the corpus.

## Strict grounding rules
1. Every factual claim in your answer MUST be traceable to at least one of \
the supplied excerpts.  Cite sources inline as [N] where N is the excerpt \
number.
2. Do NOT use knowledge from your training data.  If the excerpts do not \
contain sufficient information to answer the question, say:
   "The indexed corpus does not contain sufficient evidence to answer this \
question.  Consider ingesting additional papers on this topic."
3. If excerpts contradict each other, acknowledge the contradiction explicitly \
and cite both sides.
4. Do not speculate, extrapolate, or infer beyond what the text states.

## Output format
Write 2–4 concise paragraphs of plain prose (no markdown headers, no bullet \
lists).  Begin with a direct answer to the question, then elaborate with \
supporting detail and citations.  End with a brief sentence noting any \
important limitations visible in the evidence.

## Evidence quality legend (provided in each excerpt header)
- DIRECT — the excerpt contains explicit statistical results, clinical \
findings, or direct confirmations relevant to the query.
- INDIRECT — the excerpt is topically related but does not directly confirm \
the query.
- CONTRADICTORY — the excerpt contains language suggesting refutation or \
failure of the hypothesis.
"""


# ─── Synthesizer ─────────────────────────────────────────────────────────────

class ClaudeSynthesizer(AnswerSynthesizer):
    """
    Replaces _build_answer() with a Claude API call.

    The reasoning chain, confidence scoring, and knowledge-gap step are
    inherited unchanged from AnswerSynthesizer — only the prose answer is
    LLM-generated.
    """

    MODEL = "claude-sonnet-4-6"
    MAX_TOKENS = 1500

    def __init__(self, model: str = MODEL, max_tokens: int = MAX_TOKENS) -> None:
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self.last_prompt: dict | None = None  # populated after each call
        self._client = None  # lazy-initialised

    @property
    def _anthropic_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "anthropic package is required for ClaudeSynthesizer. "
                    "Install it with: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    # ── Public override ───────────────────────────────────────────────────────

    def synthesize(
        self,
        query_analysis: dict,
        evidence: list[EvidenceNode],
        gaps: list[str],
    ) -> tuple[str, list[ReasoningStep], float]:
        """
        Returns (answer_text, reasoning_chain, confidence_score).

        Identical contract to the parent; only _build_answer is replaced by a
        Claude call.  If the API call fails, falls back to the parent's
        rule-based answer and appends a note to the reasoning chain.
        """
        reasoning: list[ReasoningStep] = []
        step = 1

        # Step 1: Query interpretation (identical to parent)
        reasoning.append(ReasoningStep(
            step_number=step,
            label="Query interpretation",
            content=(
                f"Query classified as '{query_analysis['intent']}' intent. "
                f"Key entities identified: "
                f"{', '.join(query_analysis['entities']) if query_analysis['entities'] else 'none explicitly named'}. "
                f"Search performed with {len(query_analysis['expanded_tokens'])} expanded tokens."
            ),
            confidence=1.0,
        ))
        step += 1

        if not evidence:
            reasoning.append(ReasoningStep(
                step_number=step,
                label="Evidence retrieval",
                content="No relevant evidence found in the indexed corpus.",
                confidence=0.0,
            ))
            return (
                "I could not find relevant information in the indexed documents to answer this query. "
                "Please ensure relevant documents have been added to the corpus.",
                reasoning,
                0.0,
            )

        # Step 2: Evidence assessment (identical to parent)
        direct_count = sum(1 for e in evidence if e.support_type == "direct")
        indirect_count = sum(1 for e in evidence if e.support_type == "indirect")
        contra_count = sum(1 for e in evidence if e.support_type == "contradictory")

        reasoning.append(ReasoningStep(
            step_number=step,
            label="Evidence assessment",
            content=(
                f"Retrieved {len(evidence)} evidence node(s): "
                f"{direct_count} direct, {indirect_count} indirect, {contra_count} contradictory. "
                f"Sources span {len(set(e.doc_title for e in evidence))} document(s)."
            ),
            confidence=min(1.0, (direct_count * 0.3 + indirect_count * 0.15) / max(len(evidence), 1) + 0.4),
        ))
        step += 1

        # Step 3: LLM synthesis
        try:
            answer_text = self._call_claude(query_analysis["original"], evidence, gaps)
            synthesis_note = (
                f"Answer generated by {self.model} from {len(evidence)} evidence excerpt(s). "
                f"Primary source: '{evidence[0].doc_title}' (section: {evidence[0].section})."
                + (f" Contradictory evidence present — flagged in answer." if contra_count > 0 else "")
            )
        except Exception as exc:
            # Graceful fallback: use parent's rule-based builder
            answer_parts = self._build_answer(query_analysis, evidence)
            answer_text = "\n\n".join(answer_parts)
            synthesis_note = f"LLM synthesis failed ({exc}); used rule-based fallback."

        reasoning.append(ReasoningStep(
            step_number=step,
            label="Answer synthesis (LLM)",
            content=synthesis_note,
            confidence=self._compute_synthesis_confidence(evidence),
        ))
        step += 1

        # Step 4: Gap acknowledgment (identical to parent)
        if gaps:
            reasoning.append(ReasoningStep(
                step_number=step,
                label="Knowledge gap analysis",
                content=f"Identified {len(gaps)} gap(s): {'; '.join(gaps[:2])}.",
                confidence=0.7,
            ))
            step += 1

        confidence = self._compute_overall_confidence(evidence, gaps, direct_count)
        return answer_text, reasoning, confidence

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_user_message(
        self,
        query: str,
        evidence: list[EvidenceNode],
        gaps: list[str],
    ) -> str:
        """Format evidence nodes into the numbered-excerpt user message."""
        lines: list[str] = [
            f"## Research question\n{query}\n",
            "## Evidence excerpts",
        ]

        for i, e in enumerate(evidence, 1):
            support_label = e.support_type.upper()
            excerpt = TextProcessor.truncate_at_sentence(e.excerpt, 500)
            lines.append(
                f"\n[{i}] **{e.doc_title}** | Section: {e.section} | "
                f"Relevance: {e.relevance_score:.0%} | Type: {support_label}\n"
                f"Excerpt: {excerpt}"
            )

        if gaps:
            lines.append("\n## Known knowledge gaps in the corpus")
            for gap in gaps:
                lines.append(f"- {gap}")

        lines.append(
            "\n## Instruction\n"
            "Using ONLY the excerpts above, answer the research question. "
            "Cite sources as [N].  If the excerpts are insufficient, say so explicitly."
        )

        return "\n".join(lines)

    def _call_claude(
        self,
        query: str,
        evidence: list[EvidenceNode],
        gaps: list[str],
    ) -> str:
        """Send evidence to Claude and return the generated answer text."""
        user_message = self._build_user_message(query, evidence, gaps)

        # Store full prompt for inspection
        self.last_prompt = {
            "system": SYSTEM_PROMPT,
            "user": user_message,
        }

        response = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # prompt caching
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

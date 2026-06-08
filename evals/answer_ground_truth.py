"""
Ground-truth answer claims for the BioRAG answer-quality eval (LLM-as-judge).

Each AnswerClaim ties to a RetrievalQuery by query_id and specifies what a
correct, complete answer must assert.  The five fields map directly to the
five rubric dimensions scored by the judge:

  reference_claim    → Semantic Coverage
  expected_entities  → Entity Coverage
  expected_direction → Directional Agreement
  expected_context   → Contextual Accuracy
  (quantitative_detail is inferred from the reference_claim text)

Coverage: 10 claims across all four corpus documents.
  Q01–Q03, Q07 — Alzheimer's disease / neuro_2026_001
  Q08, Q09     — Resveratrol / cardio_2026_001
  Q11, Q12     — EGFR-mutant NSCLC / onco_2026_001
  Q13, Q14     — Carbapenem-resistant K. pneumoniae / infect_2026_001
"""

from dataclasses import dataclass, field


@dataclass
class AnswerClaim:
    """Reference claim for one eval query."""
    query_id: str
    reference_claim: str
    expected_entities: list[str] = field(default_factory=list)
    expected_direction: str = ""
    expected_context: str = ""
    notes: str = ""


# ── Alzheimer's Disease Claims ────────────────────────────────────────────────

ALZHEIMER_CLAIMS: list[AnswerClaim] = [
    AnswerClaim(
        query_id="Q01",
        reference_claim=(
            "Blood-based biomarkers — including plasma amyloid-beta ratio "
            "(Aβ42/Aβ40) and phosphorylated tau isoforms (p-tau217, p-tau181) — "
            "can detect Alzheimer's disease pathology before clinical symptoms appear."
        ),
        expected_entities=["amyloid-beta", "p-tau217", "p-tau181", "Aβ42/Aβ40"],
        expected_direction="elevated (p-tau) and decreased (Aβ42/Aβ40 ratio) in preclinical AD",
        expected_context="preclinical / pre-symptomatic Alzheimer's disease",
        notes="Tests whether the answer names both amyloid and tau markers specifically.",
    ),
    AnswerClaim(
        query_id="Q02",
        reference_claim=(
            "The plasma Aβ42/Aβ40 ratio shows high accuracy for predicting "
            "amyloid pathology and performs comparably to CSF measurements and "
            "PET imaging in identifying Alzheimer's disease."
        ),
        expected_entities=["Aβ42/Aβ40", "amyloid", "CSF", "PET"],
        expected_direction="decreased Aβ42/Aβ40 ratio indicates amyloid pathology",
        expected_context="predicting amyloid pathology / Alzheimer's diagnosis",
        notes=(
            "Key test: does the answer state that the ratio DECREASES (not increases) "
            "and compare it explicitly to CSF/PET?"
        ),
    ),
    AnswerClaim(
        query_id="Q03",
        reference_claim=(
            "Phosphorylated tau, particularly p-tau217 and p-tau181, demonstrates "
            "high diagnostic accuracy in blood for detecting Alzheimer's disease, "
            "with AUC values typically above 0.85."
        ),
        expected_entities=["p-tau217", "p-tau181", "AUC"],
        expected_direction="elevated p-tau in Alzheimer's disease",
        expected_context="blood-based diagnosis of Alzheimer's disease",
        notes="Checks for both isoforms and a quantitative AUC threshold.",
    ),
    AnswerClaim(
        query_id="Q07",
        reference_claim=(
            "Blood-based biomarkers such as plasma amyloid-beta and "
            "phosphorylated tau enable early detection of preclinical Alzheimer's "
            "disease, providing a window for therapeutic intervention before "
            "significant neuronal loss occurs."
        ),
        expected_entities=["amyloid-beta", "phosphorylated tau", "preclinical"],
        expected_direction="early detection enables intervention before neuronal loss",
        expected_context="preclinical Alzheimer's disease, pre-symptom intervention window",
        notes="Framed as a treatment-enablement query; checks clinical actionability framing.",
    ),
]

# ── Cardiology / Diabetes Claims ──────────────────────────────────────────────

CARDIO_CLAIMS: list[AnswerClaim] = [
    AnswerClaim(
        query_id="Q08",
        reference_claim=(
            "Resveratrol supplementation significantly reduces serum uric acid "
            "in type 2 diabetes patients, with a pooled weighted mean difference "
            "of −0.42 mg/dL (P = 0.0005) and low heterogeneity (I² = 0%) across "
            "12 RCTs with 636 participants."
        ),
        expected_entities=["resveratrol", "uric acid", "WMD", "RCT"],
        expected_direction="decreased serum uric acid",
        expected_context="type 2 diabetes patients, meta-analysis of RCTs",
        notes=(
            "The WMD −0.42 and I²=0% are explicit numbers in the corpus; "
            "directional and quantitative dimensions should both be scoreable."
        ),
    ),
    AnswerClaim(
        query_id="Q09",
        reference_claim=(
            "Resveratrol supplementation has no significant effect on serum "
            "creatinine (SMD: 0.05) or blood urea nitrogen (WMD: −0.01) in "
            "type 2 diabetes patients, regardless of RSV dose or intervention duration."
        ),
        expected_entities=["resveratrol", "serum creatinine", "blood urea nitrogen", "BUN"],
        expected_direction="no significant effect on SCr or BUN",
        expected_context="type 2 diabetes with diabetic kidney disease, subgroup analyses",
        notes=(
            "Direction here is 'null effect'; tests whether the answer correctly "
            "states non-significance rather than improvement."
        ),
    ),
]

# ── Oncology Claims ───────────────────────────────────────────────────────────

ONCO_CLAIMS: list[AnswerClaim] = [
    AnswerClaim(
        query_id="Q11",
        reference_claim=(
            "In EGFR-mutant non-small cell lung cancer with high PD-L1 "
            "expression (≥50%), EGFR tyrosine kinase inhibitors (TKIs) remain "
            "the preferred first-line therapy over immune checkpoint inhibitors "
            "(ICIs) due to superior efficacy in this molecular subgroup."
        ),
        expected_entities=["EGFR", "PD-L1", "TKI", "ICI", "NSCLC"],
        expected_direction="TKI preferred over ICI",
        expected_context="EGFR-mutant NSCLC with high PD-L1 expression (≥50%), first-line",
        notes="Tests whether the answer handles the co-occurrence of two biomarkers correctly.",
    ),
    AnswerClaim(
        query_id="Q12",
        reference_claim=(
            "EGFR tyrosine kinase inhibitors show superior progression-free "
            "survival compared to immune checkpoint inhibitors in EGFR-mutant "
            "NSCLC, irrespective of PD-L1 expression level."
        ),
        expected_entities=["EGFR TKI", "ICI", "progression-free survival", "PD-L1"],
        expected_direction="TKI superior to ICI for progression-free survival",
        expected_context="EGFR-mutant NSCLC, regardless of PD-L1 level",
        notes="The 'regardless of PD-L1' qualifier is the contextual test here.",
    ),
]

# ── Infectious Disease Claims ─────────────────────────────────────────────────

INFECT_CLAIMS: list[AnswerClaim] = [
    AnswerClaim(
        query_id="Q13",
        reference_claim=(
            "Carbapenem-resistant Klebsiella pneumoniae co-producing KPC-2 and "
            "NDM-1 (K2N1-CRKP) is a serious clinical challenge; eravacycline "
            "is among the active treatment options, demonstrating efficacy in "
            "both in vitro and in vivo models."
        ),
        expected_entities=["eravacycline", "KPC-2", "NDM-1", "carbapenem-resistant", "K. pneumoniae"],
        expected_direction="eravacycline active against K2N1-CRKP",
        expected_context="co-producing KPC-2 and NDM-1 carbapenem-resistant K. pneumoniae",
        notes="Tests whether both resistance mechanisms (KPC-2 and NDM-1) are named.",
    ),
    AnswerClaim(
        query_id="Q14",
        reference_claim=(
            "Eravacycline demonstrates in vitro antimicrobial activity against "
            "KPC-2 and NDM-1 co-producing carbapenem-resistant K. pneumoniae, "
            "and shows in vivo efficacy in animal infection models, supporting "
            "its potential as a treatment option."
        ),
        expected_entities=["eravacycline", "KPC-2", "NDM-1", "MIC", "in vitro", "in vivo"],
        expected_direction="eravacycline effective both in vitro and in vivo",
        expected_context="KPC-2 and NDM-1 co-producing CRKP, in vitro and animal models",
        notes=(
            "Both in vitro and in vivo context are required for full contextual accuracy; "
            "MIC is the quantitative detail expected."
        ),
    ),
]

# ── Full claim set ────────────────────────────────────────────────────────────

ANSWER_CLAIMS: list[AnswerClaim] = (
    ALZHEIMER_CLAIMS
    + CARDIO_CLAIMS
    + ONCO_CLAIMS
    + INFECT_CLAIMS
)

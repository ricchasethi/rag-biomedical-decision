"""
Ground-truth relevance judgements for the BioRAG retrieval eval.

Each RetrievalQuery entry defines:
  - query_id   : stable identifier so results are traceable across runs
  - query      : natural-language question sent to the retrieval pipeline
  - intent     : expected QueryAnalyzer intent label (used for stratified reporting)
  - relevant_docs : {doc_id: grade}
        2 = highly relevant — the document directly answers the question
        1 = partially relevant — the document contains related but tangential content
        absent entry means grade 0 (irrelevant)

Corpus doc IDs in the sample corpus:
  cardio_2026_001  — Resveratrol & renal biomarkers in T2DM (RCT meta-analysis)
  onco_2026_001    — EGFR-mutant NSCLC with high PD-L1: TKI vs immunotherapy
  infect_2026_001  — Eravacycline vs KPC-2/NDM-1 carbapenem-resistant K. pneumoniae
  neuro_2026_001   — Blood-based biomarkers (amyloid-β, tau) for Alzheimer's disease

Alzheimer's queries (Q01–Q07) are the primary eval focus.
Mixed / cross-topic queries (Q08–Q16) test inter-document discrimination.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    query: str
    intent: str
    relevant_docs: dict[str, int] = field(default_factory=dict)


# ── Alzheimer's Disease Queries (primary focus) ───────────────────────────────

ALZHEIMER_QUERIES: list[RetrievalQuery] = [
    RetrievalQuery(
        query_id="Q01",
        query="What blood-based biomarkers can detect Alzheimer's disease before clinical diagnosis?",
        intent="diagnosis",
        relevant_docs={
            "neuro_2026_001": 2,   # umbrella review of BBMs for preclinical AD detection
        },
    ),
    RetrievalQuery(
        query_id="Q02",
        query="How accurate is plasma amyloid-beta measurement for predicting Alzheimer's pathology?",
        intent="diagnosis",
        relevant_docs={
            "neuro_2026_001": 2,   # covers plasma Aβ42/Aβ40 ratio diagnostic performance
        },
    ),
    RetrievalQuery(
        query_id="Q03",
        query="What is the diagnostic performance of phosphorylated tau in blood for Alzheimer's disease?",
        intent="diagnosis",
        relevant_docs={
            "neuro_2026_001": 2,   # p-tau217 and p-tau181 are key BBMs reviewed
        },
    ),
    RetrievalQuery(
        query_id="Q04",
        query="Can blood biomarkers replace CSF analysis or PET imaging in Alzheimer's disease workup?",
        intent="comparison",
        relevant_docs={
            "neuro_2026_001": 2,   # directly addresses BBM vs CSF/PET in review
        },
    ),
    RetrievalQuery(
        query_id="Q05",
        query="What does systematic review evidence show about tau pathology biomarkers in prodromal Alzheimer's?",
        intent="prognosis",
        relevant_docs={
            "neuro_2026_001": 2,   # umbrella review of systematic reviews on tau BBMs
        },
    ),
    RetrievalQuery(
        query_id="Q06",
        query="How do amyloid plaques and tau tangles manifest as measurable blood signals in early Alzheimer's?",
        intent="mechanism",
        relevant_docs={
            "neuro_2026_001": 2,   # hallmarks of AD and their blood-based detection are central topic
        },
    ),
    RetrievalQuery(
        query_id="Q07",
        query="What biomarkers enable timely intervention in preclinical Alzheimer's disease?",
        intent="treatment",
        relevant_docs={
            "neuro_2026_001": 2,   # early detection enabling intervention is stated objective
        },
    ),
]

# ── Cardiology / Diabetes Queries ─────────────────────────────────────────────

CARDIO_QUERIES: list[RetrievalQuery] = [
    RetrievalQuery(
        query_id="Q08",
        query="Does resveratrol supplementation reduce uric acid levels in type 2 diabetes patients?",
        intent="treatment",
        relevant_docs={
            "cardio_2026_001": 2,  # primary finding: RSV reduces serum UA (WMD −0.42)
        },
    ),
    RetrievalQuery(
        query_id="Q09",
        query="What is the effect of resveratrol on serum creatinine and blood urea nitrogen in diabetic kidney disease?",
        intent="treatment",
        relevant_docs={
            "cardio_2026_001": 2,  # explicitly reports BUN and SCr outcomes (no significant effect)
        },
    ),
    RetrievalQuery(
        query_id="Q10",
        query="What biomarkers reflect renal injury in patients with type 2 diabetes mellitus?",
        intent="diagnosis",
        relevant_docs={
            "cardio_2026_001": 2,  # BUN, SCr, serum UA as renal injury markers
            "neuro_2026_001": 1,   # general biomarker review — tangential
        },
    ),
]

# ── Oncology Queries ──────────────────────────────────────────────────────────

ONCO_QUERIES: list[RetrievalQuery] = [
    RetrievalQuery(
        query_id="Q11",
        query="How does high PD-L1 expression affect treatment decisions in EGFR-mutant non-small cell lung cancer?",
        intent="treatment",
        relevant_docs={
            "onco_2026_001": 2,   # central topic: EGFR-mutant NSCLC + high PD-L1 expression
        },
    ),
    RetrievalQuery(
        query_id="Q12",
        query="What is the role of EGFR tyrosine kinase inhibitors versus immune checkpoint inhibitors in NSCLC?",
        intent="comparison",
        relevant_docs={
            "onco_2026_001": 2,   # TKI vs ICI in EGFR-mutant NSCLC is the key comparison
        },
    ),
]

# ── Infectious Disease Queries ────────────────────────────────────────────────

INFECT_QUERIES: list[RetrievalQuery] = [
    RetrievalQuery(
        query_id="Q13",
        query="What are the treatment options for carbapenem-resistant Klebsiella pneumoniae infections?",
        intent="treatment",
        relevant_docs={
            "infect_2026_001": 2,  # K2N1-CRKP treatment is the subject
        },
    ),
    RetrievalQuery(
        query_id="Q14",
        query="How effective is eravacycline against KPC-2 and NDM-1 co-producing bacteria?",
        intent="treatment",
        relevant_docs={
            "infect_2026_001": 2,  # eravacycline vs K2N1-CRKP in vitro and in vivo
        },
    ),
]

# ── Cross-Topic / Discrimination Queries ──────────────────────────────────────

CROSS_QUERIES: list[RetrievalQuery] = [
    RetrievalQuery(
        query_id="Q15",
        query="What systematic review evidence exists for biomarker-guided therapy in complex diseases?",
        intent="general",
        relevant_docs={
            "neuro_2026_001": 2,   # umbrella review of biomarker systematic reviews
            "cardio_2026_001": 1,  # RCT meta-analysis (biomarker-guided in diabetes)
            "onco_2026_001": 1,    # biomarker-guided therapy (EGFR/PD-L1) in NSCLC
        },
    ),
    RetrievalQuery(
        query_id="Q16",
        query="What is the prevalence of drug-resistant bacterial infections and their clinical management?",
        intent="epidemiology",
        relevant_docs={
            "infect_2026_001": 2,  # epidemiology of CRKP + management
        },
    ),
]

# ── Full eval set ─────────────────────────────────────────────────────────────

EVAL_QUERIES: list[RetrievalQuery] = (
    ALZHEIMER_QUERIES
    + CARDIO_QUERIES
    + ONCO_QUERIES
    + INFECT_QUERIES
    + CROSS_QUERIES
)

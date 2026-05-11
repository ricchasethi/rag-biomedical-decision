"""
BioRAG Decision-Support Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core retrieval-augmented generation engine for scientific/biomedical documents.
Uses TF-IDF + BM25-style retrieval with a layered reasoning pipeline.
"""

import re
import math
import json
import hashlib
import textwrap
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    text: str
    section: str
    page: int
    tokens: list[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    match_terms: list[str] = field(default_factory=list)


@dataclass
class EvidenceNode:
    source_id: str
    doc_title: str
    section: str
    excerpt: str
    relevance_score: float
    support_type: str   # "direct" | "indirect" | "contradictory"
    key_terms: list[str] = field(default_factory=list)


@dataclass
class ReasoningStep:
    step_number: int
    label: str
    content: str
    confidence: float


@dataclass
class DecisionOutput:
    query: str
    answer: str
    confidence: float
    confidence_label: str
    evidence: list[EvidenceNode]
    reasoning_chain: list[ReasoningStep]
    knowledge_gaps: list[str]
    follow_up_questions: list[str]
    sources_used: int
    total_chunks_searched: int


# ─── Text Processing ─────────────────────────────────────────────────────────

class TextProcessor:
    """Biomedical-aware tokenizer and cleaner."""

    # Common biomedical stopwords that don't add retrieval value
    STOPWORDS = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "by","from","up","about","into","through","during","is","are","was",
        "were","be","been","being","have","has","had","do","does","did","will",
        "would","could","should","may","might","shall","can","this","that",
        "these","those","it","its","they","them","their","we","our","you","your",
        "he","she","his","her","i","my","me","us","which","who","whom","when",
        "where","why","how","what","all","each","both","few","more","most","other",
        "some","such","no","not","only","same","so","than","then","there","too",
        "very","just","also","even","much","well","back","still","way","per",
    }

    # Biomedical abbreviation expansions for better matching
    ABBREV_MAP = {
        "dna": "deoxyribonucleic acid dna",
        "rna": "ribonucleic acid rna",
        "mrna": "messenger ribonucleic acid mrna",
        "mirna": "micro ribonucleic acid mirna",
        "pcr": "polymerase chain reaction pcr",
        "elisa": "enzyme linked immunosorbent assay elisa",
        "ct": "computed tomography ct scan",
        "mri": "magnetic resonance imaging mri",
        "ecg": "electrocardiogram ecg ekg",
        "bp": "blood pressure bp",
        "hr": "heart rate hr",
        "bmi": "body mass index bmi",
        "auc": "area under curve auc",
        "ci": "confidence interval ci",
        "rr": "relative risk rr risk ratio",
        "or": "odds ratio or",
        "hr_stat": "hazard ratio",
        "p53": "tumor protein p53 tp53",
        "vegf": "vascular endothelial growth factor vegf",
        "tnf": "tumor necrosis factor tnf",
        "il": "interleukin",
    }

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Tokenize with biomedical awareness."""
        text = text.lower()
        # Preserve hyphenated compounds (e.g., anti-inflammatory)
        text = re.sub(r'(\w)-(\w)', r'\1_\2', text)
        # Keep alphanumeric + underscores
        tokens = re.findall(r'\b[a-z0-9_]{2,}\b', text)
        # Restore hyphens for display
        tokens = [t.replace('_', '-') for t in tokens]
        # Remove stopwords but keep short biomedical terms (2-3 chars like IL-6)
        filtered = []
        for t in tokens:
            if t not in cls.STOPWORDS or len(t) <= 3:
                filtered.append(t)
        return filtered

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Normalize whitespace and remove citation artifacts from biomedical text.

        Five citation forms are stripped, covering the two encoding styles
        found in PMC full-text and PubMed abstract sources:

        No-space style (JATS <xref> artifacts after XML concatenation):
          .1 / .7,8    "DKD.1 Within"           → "DKD. Within"
          ,3 / ,12,13  "anti-inflammatory,3 hypo" → "anti-inflammatory hypo"

        Spaced style (citations typeset with surrounding spaces):
          . N / ) N    "worldwide. 1 Its"        → "worldwide. Its"
                       "(PRS) 9 and"             → "(PRS) and"
          . N , M      "tau. 2 , 3 Despite"      → "tau. Despite"
          N , M , P…   " 31 , 32 , 33 In AD"     → " In AD"  (3+ numbers)
          , N          "SBayesRC, 37 and"         → "SBayesRC and"

        Intentionally NOT stripped — too ambiguous without NLP context:
          word N word  "eQTLGen 38 yielded" or "50 million individuals"
                       (single inline number after a word, no punctuation cue)
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)   # fix line-break hyphenation
        text = re.sub(r'\[\d+\]', '', text)               # [1] bracket citations

        # ── No-space style ────────────────────────────────────────────────────
        # Period-superscript: letter/% + .N (preceding digit excluded to protect
        # decimals like 0.05 and CI values like −0.66).
        text = re.sub(
            r'(?<=[A-Za-z%])\.\d{1,3}(?:,\d{1,3})*(?=\s+[A-Za-z])',
            '.',
            text,
        )
        # Comma-superscript (no space): word,N word
        text = re.sub(
            r'(?<=[A-Za-z])(,\d{1,3})+(?=\s+[a-z])',
            '',
            text,
        )

        # ── Spaced style ──────────────────────────────────────────────────────
        # After sentence-ending . or closing ): ". 1 Word"  ") 9 and"
        text = re.sub(
            r'(?<=[.)]) \d{1,3}(?: , \d{1,3})* (?=[A-Za-z])',
            ' ',
            text,
        )
        # Standalone series of 3+ spaced numbers: " 31 , 32 , 33 Word"
        text = re.sub(
            r' \d{1,3}(?: , \d{1,3}){2,} (?=[A-Za-z])',
            ' ',
            text,
        )
        # Comma-space citation: "word, 37 and"
        text = re.sub(
            r'(?<=[A-Za-z]), \d{1,3}(?: , \d{1,3})* (?=[a-z])',
            ' ',
            text,
        )

        return re.sub(r'\s+', ' ', text).strip()  # re-normalise after removals

    @classmethod
    def extract_sentences(cls, text: str) -> list[str]:
        """Split into sentences, aware of abbreviations."""
        # Don't split on common abbreviations
        abbrevs = r'(?:et al|e\.g|i\.e|vs|Dr|Mr|Ms|Prof|Fig|Tab|Eq|approx|avg|max|min|std|SD|SE)'
        protected = re.sub(f'({abbrevs})\\.', r'\1<DOT>', text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
        return [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]

    @classmethod
    def truncate_at_sentence(cls, text: str, max_chars: int) -> str:
        """Return text truncated at a sentence boundary not exceeding max_chars.

        Falls back to the raw slice only when no sentence fits within the limit.
        """
        if len(text) <= max_chars:
            return text
        result = ""
        for sentence in cls.extract_sentences(text):
            candidate = f"{result} {sentence}".strip() if result else sentence
            if len(candidate) > max_chars:
                break
            result = candidate
        return result if result else text[:max_chars]


# ─── Document Chunker ─────────────────────────────────────────────────────────

class DocumentChunker:
    """
    Semantic chunker that respects document structure.
    Uses overlapping windows to avoid splitting context across chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 80,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.processor = TextProcessor()

    def _detect_section(self, text: str) -> str:
        """Heuristically detect document section from text headers."""
        patterns = [
            (r'(?i)^(abstract)\b', 'Abstract'),
            (r'(?i)^(introduction|background)\b', 'Introduction'),
            (r'(?i)^(method|methodology|materials?\s+and\s+method)', 'Methods'),
            (r'(?i)^(result)', 'Results'),
            (r'(?i)^(discussion)', 'Discussion'),
            (r'(?i)^(conclusion)', 'Conclusion'),
            (r'(?i)^(reference|bibliography)', 'References'),
            (r'(?i)^(supplementary|appendix)', 'Supplementary'),
        ]
        first_line = text.split('\n')[0].strip()
        for pattern, section in patterns:
            if re.match(pattern, first_line):
                return section
        return 'Body'

    def chunk_document(self, doc_id: str, title: str, text: str) -> list[Chunk]:
        """Split document into semantically coherent, overlapping chunks."""
        text = self.processor.clean_text(text)
        sentences = self.processor.extract_sentences(text)

        chunks = []
        current_sentences = []
        current_len = 0
        section = 'Body'
        char_pos = 0

        for sent in sentences:
            sent_len = len(sent)

            # Detect section header
            new_section = self._detect_section(sent)
            if new_section != 'Body':
                section = new_section

            # If adding this sentence exceeds chunk size, flush
            if current_len + sent_len > self.chunk_size and current_sentences:
                chunk_text = ' '.join(current_sentences)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_id = hashlib.md5(
                        f"{doc_id}:{char_pos}".encode()
                    ).hexdigest()[:12]
                    chunk = Chunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        doc_title=title,
                        text=chunk_text,
                        section=section,
                        page=max(1, char_pos // 2500 + 1),
                        char_start=char_pos - current_len,
                        char_end=char_pos,
                    )
                    chunk.tokens = self.processor.tokenize(chunk_text)
                    chunks.append(chunk)

                # Overlap: keep last N characters worth of sentences
                overlap_text = ''
                overlap_sents = []
                for s in reversed(current_sentences):
                    if len(overlap_text) + len(s) > self.chunk_overlap:
                        break
                    overlap_text = s + ' ' + overlap_text
                    overlap_sents.insert(0, s)

                current_sentences = overlap_sents
                current_len = sum(len(s) for s in current_sentences)

            current_sentences.append(sent)
            current_len += sent_len
            char_pos += sent_len + 1

        # Flush remaining
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            if len(chunk_text) >= self.min_chunk_size:
                chunk_id = hashlib.md5(
                    f"{doc_id}:{char_pos}".encode()
                ).hexdigest()[:12]
                chunk = Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    doc_title=title,
                    text=chunk_text,
                    section=section,
                    page=max(1, char_pos // 2500 + 1),
                    char_start=char_pos - current_len,
                    char_end=char_pos,
                )
                chunk.tokens = self.processor.tokenize(chunk_text)
                chunks.append(chunk)

        return chunks


# ─── Index ────────────────────────────────────────────────────────────────────

class InvertedIndex:
    """
    BM25-based inverted index for efficient retrieval.
    BM25 (Okapi BM25) outperforms TF-IDF for biomedical texts.
    """

    # BM25 hyperparameters
    K1 = 1.5   # term frequency saturation
    B = 0.75   # document length normalization

    def __init__(self):
        self.index: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.chunks: dict[str, Chunk] = {}
        self.doc_lengths: dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0
        self.df: dict[str, int] = Counter()   # document frequency per term

    def add_chunk(self, chunk: Chunk) -> None:
        """Index a chunk."""
        self.chunks[chunk.id] = chunk
        self.doc_lengths[chunk.id] = len(chunk.tokens)

        term_freq = Counter(chunk.tokens)
        for term, freq in term_freq.items():
            self.index[term].append((chunk.id, freq))
            self.df[term] += 1

        self.doc_count += 1
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / self.doc_count

    def _bm25_score(self, term: str, chunk_id: str, tf: int) -> float:
        """Compute BM25 score for a term in a document."""
        if self.doc_count == 0 or term not in self.df:
            return 0.0

        # IDF component (with smoothing)
        n = self.doc_count
        df = self.df[term]
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

        # TF normalization
        dl = self.doc_lengths.get(chunk_id, self.avg_doc_length)
        tf_norm = (tf * (self.K1 + 1)) / (
            tf + self.K1 * (1 - self.B + self.B * dl / self.avg_doc_length)
        )

        return idf * tf_norm

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[RetrievedChunk]:
        """BM25 search over indexed chunks."""
        scores: dict[str, float] = defaultdict(float)
        chunk_match_terms: dict[str, list[str]] = defaultdict(list)

        for term in set(query_tokens):
            if term not in self.index:
                # Prefix matching only when the query token is long enough that
                # a 4-char prefix is discriminative. Short tokens (≤6 chars) like
                # "hemo", "age", "il" would match too many unrelated indexed terms.
                if len(term) > 6:
                    matches = [t for t in self.index if t.startswith(term[:4]) and len(t) >= 4]
                    for matched_term in matches[:3]:
                        for chunk_id, tf in self.index[matched_term]:
                            scores[chunk_id] += self._bm25_score(matched_term, chunk_id, tf) * 0.7
                            chunk_match_terms[chunk_id].append(f"~{matched_term}")
            else:
                for chunk_id, tf in self.index[term]:
                    scores[chunk_id] += self._bm25_score(term, chunk_id, tf)
                    chunk_match_terms[chunk_id].append(term)

        # Sort by score
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (chunk_id, score) in enumerate(sorted_chunks, start=1):
            chunk = self.chunks[chunk_id]
            results.append(RetrievedChunk(
                chunk=chunk,
                score=score,
                rank=rank,
                match_terms=chunk_match_terms[chunk_id][:8],
            ))

        return results

    def stats(self) -> dict:
        return {
            "total_chunks": self.doc_count,
            "unique_terms": len(self.index),
            "avg_chunk_length": round(self.avg_doc_length, 1),
        }


# ─── Query Analyzer ───────────────────────────────────────────────────────────

class QueryAnalyzer:
    """
    Parses and enriches queries with biomedical intent detection.
    Classifies query type to guide downstream reasoning.
    """

    QUERY_TYPES = {
        "mechanism": ["how does", "mechanism", "pathway", "process", "work"],
        "comparison": ["compare", "difference", "versus", "vs", "better", "worse", "superior"],
        "causation": ["cause", "why", "reason", "lead to", "result in", "due to"],
        "treatment": ["treat", "therapy", "intervention", "drug", "dose", "efficacy"],
        "diagnosis": ["diagnose", "detect", "identify", "marker", "criterion", "symptom"],
        "prognosis": ["prognosis", "survival", "outcome", "predict", "mortality", "recurrence"],
        "epidemiology": ["prevalence", "incidence", "population", "risk factor", "epidemiology"],
        "definition": ["what is", "define", "definition", "describe"],
    }

    def __init__(self):
        self.processor = TextProcessor()

    def analyze(self, query: str) -> dict:
        """Extract intent, entities, and expanded tokens from query."""
        query_lower = query.lower()
        tokens = self.processor.tokenize(query)

        # Classify intent
        intent = "general"
        for qtype, patterns in self.QUERY_TYPES.items():
            if any(p in query_lower for p in patterns):
                intent = qtype
                break

        # Detect key entities (capitalized multi-word terms, gene names, drug names).
        # Exclude interrogative and auxiliary words that are capitalised only because
        # they open the sentence — these add no retrieval signal and pollute BM25 scoring.
        _ENTITY_BLOCKLIST = {
            "What","Which","Who","Whom","Whose","How","Why","When","Where",
            "Is","Are","Was","Were","Do","Does","Did","Can","Could","Should",
            "Would","Will","Shall","May","Might","Have","Has","Had",
            "The","A","An","This","That","These","Those",
        }
        entities = [
            e for e in re.findall(r'\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*\b', query)
            if e not in _ENTITY_BLOCKLIST
        ]

        # Expand abbreviations
        expanded_tokens = list(tokens)
        for token in tokens:
            if token.lower() in self.processor.ABBREV_MAP:
                expansion_tokens = self.processor.tokenize(
                    self.processor.ABBREV_MAP[token.lower()]
                )
                expanded_tokens.extend(expansion_tokens)

        return {
            "original": query,
            "tokens": tokens,
            "expanded_tokens": list(set(expanded_tokens)),
            "intent": intent,
            "entities": entities,
            "is_comparison": intent == "comparison",
            "needs_quantitative": intent in {"epidemiology", "prognosis", "treatment"},
        }


# ─── Reranker ─────────────────────────────────────────────────────────────────

class Reranker:
    """
    Re-ranks initial BM25 results using additional signals:
    - Section relevance (Methods/Results prioritized for factual queries)
    - Term density (concentration of query terms)
    - Discriminative-token recall (penalises off-topic false positives)
    """

    SECTION_WEIGHTS = {
        "mechanism": {"Results": 1.3, "Methods": 1.1, "Discussion": 1.0, "Abstract": 0.9},
        "comparison": {"Results": 1.4, "Discussion": 1.2, "Abstract": 1.0},
        "treatment":  {"Results": 1.4, "Methods": 1.2, "Discussion": 1.0},
        "diagnosis":  {"Results": 1.3, "Methods": 1.2, "Abstract": 1.0},
        "definition": {"Introduction": 1.3, "Abstract": 1.2, "Body": 1.0},
        "general":    {"Abstract": 1.1, "Results": 1.1, "Discussion": 1.0},
    }

    # Terms that appear in virtually every biomedical paper and therefore carry
    # no discriminative signal about *which* disease or entity a paper covers.
    # A chunk that matches only these terms but misses the specific entity token
    # (e.g. "alzheimer", "egfr") gets a heavy penalty so it does not crowd out
    # truly relevant results.
    #
    # Includes both domain nouns (biomarker, disease) AND common clinical/scientific
    # verbs and adjectives that appear in any query about any disease area
    # (predict, detect, identify, assess…).  Without these, a query like
    # "What biomarkers predict Alzheimer's?" would treat "predict" as discriminative
    # and falsely rescue off-topic papers that heavily use "predict".
    _GENERIC_BIO_TERMS: frozenset[str] = frozenset({
        # Domain nouns ubiquitous across all disease areas
        "biomarker", "biomarkers", "marker", "markers", "disease", "diseases",
        "patient", "patients", "study", "studies", "treatment", "treatments",
        "expression", "level", "levels", "plasma", "blood", "serum",
        "cell", "cells", "protein", "proteins", "gene", "genes",
        "clinical", "risk", "therapy", "therapies", "analysis",
        "result", "results", "finding", "findings", "association",
        "tumor", "tumour", "sample", "data", "method", "methods",
        "diagnosis", "prognosis", "detection", "prediction", "response",
        "factor", "factors", "effect", "effects", "increase", "decrease",
        "significant", "significantly", "compared", "group", "groups",
        "control", "measure", "measurement", "outcome", "outcomes",
        "cohort", "population", "baseline", "median", "mean", "ratio",
        "age", "sex", "test", "tissue", "model", "concentration", "type",
        # Common clinical/scientific verbs used in queries about any disease
        "predict", "predicted", "predictive", "predictor", "predictors",
        "detect", "detected", "detectable", "identify", "identified",
        "assess", "assessed", "assessment", "evaluate", "evaluated",
        "measure", "measured", "determine", "determined",
        "associate", "associated", "correlate", "correlated", "correlation",
        "indicate", "indicated", "demonstrate", "demonstrated",
        "report", "reported", "investigate", "investigated",
        "suggest", "suggested", "compare", "compared",
        "improve", "improved", "reduce", "reduced", "inhibit", "inhibited",
        "use", "used", "using", "show", "shown", "found",
        # Common adjectives appearing in clinical queries across all diseases
        "early", "late", "novel", "potential", "effective", "accurate",
        "sensitive", "specific", "elevated", "increased", "decreased",
        "high", "low", "new", "current", "recent", "common",
    })

    def rerank(
        self,
        results: list[RetrievedChunk],
        query_analysis: dict,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Apply section weight, term density, and discriminative-token recall to rerank results."""
        intent = query_analysis["intent"]
        section_weights = self.SECTION_WEIGHTS.get(intent, self.SECTION_WEIGHTS["general"])
        query_tokens = set(query_analysis["expanded_tokens"])

        # Tokens that are unique to this query's topic (not generic biomedical noise).
        # Missing any of these from a chunk is a strong signal the chunk is off-topic.
        discriminative = query_tokens - self._GENERIC_BIO_TERMS

        reranked = []
        for r in results:
            section_w = section_weights.get(r.chunk.section, 1.0)

            # Term density: fraction of chunk tokens that are query tokens
            chunk_token_set = set(r.chunk.tokens)
            if chunk_token_set:
                density = len(query_tokens & chunk_token_set) / len(chunk_token_set)
            else:
                density = 0.0

            # Position bonus for early chunks (abstract/intro often most informative)
            position_bonus = 1.0 + (0.05 / max(r.rank, 1))

            # Discriminative-token recall penalty.
            # When the query has specific entity tokens (disease names, gene names, etc.)
            # a chunk that matches none of them is almost certainly an off-topic false
            # positive promoted by a generic term like "biomarker".
            if discriminative:
                matched = discriminative & chunk_token_set
                if not matched:
                    entity_factor = 0.15   # matches zero entity-specific tokens → demote hard
                else:
                    entity_factor = 0.6 + 0.4 * (len(matched) / len(discriminative))
            else:
                entity_factor = 1.0  # fully generic query — no penalty

            adjusted_score = (
                r.score * section_w * (1 + density * 0.4) * position_bonus * entity_factor
            )

            reranked.append(RetrievedChunk(
                chunk=r.chunk,
                score=adjusted_score,
                rank=r.rank,
                match_terms=r.match_terms,
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(reranked):
            r.rank = i + 1

        return reranked[:top_k]


# ─── Evidence Classifier ──────────────────────────────────────────────────────

class EvidenceClassifier:
    """
    Classifies each retrieved chunk as direct/indirect/contradictory evidence
    and extracts key biomedical terms that justify the classification.
    """

    DIRECT_PATTERNS = [
        r'\b(demonstrate|show|reveal|confirm|establish|prove|indicate|suggest)\b',
        r'\b(significant|p\s*[<=>]\s*0\.\d+|95%\s*CI|odds ratio|hazard ratio)\b',
        r'\b(result|finding|observation|data|evidence)\b.*\b(support|consistent)\b',
    ]

    CONTRADICTORY_PATTERNS = [
        r'\b(contradict|inconsistent|conflict|contrary|dispute|challenge|refute)\b',
        r'\b(however|nevertheless|in contrast|on the other hand)\b.*\b(no|not|fail)\b',
        r'\b(no significant|non-significant|failed to|did not|could not)\b',
    ]

    def classify(
        self,
        chunk: Chunk,
        query_analysis: dict,
    ) -> tuple[str, list[str]]:
        """Return (support_type, key_terms)."""
        text_lower = chunk.text.lower()

        # Check contradictory first
        for pattern in self.CONTRADICTORY_PATTERNS:
            if re.search(pattern, text_lower):
                key_terms = self._extract_key_terms(chunk.text, query_analysis)
                return "contradictory", key_terms

        # Check direct
        for pattern in self.DIRECT_PATTERNS:
            if re.search(pattern, text_lower):
                key_terms = self._extract_key_terms(chunk.text, query_analysis)
                return "direct", key_terms

        key_terms = self._extract_key_terms(chunk.text, query_analysis)
        return "indirect", key_terms

    def _extract_key_terms(self, text: str, query_analysis: dict) -> list[str]:
        """Extract biomedical entities and query-matching terms."""
        # Named entities (capitalized terms)
        entities = re.findall(
            r'\b[A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,}){0,2}\b', text
        )
        # Stats patterns
        stats = re.findall(
            r'(?:p\s*[<=>]\s*0\.\d+|OR\s*=?\s*[\d.]+|HR\s*=?\s*[\d.]+|'
            r'95%\s*CI|RR\s*=?\s*[\d.]+|\d+\.?\d*%)', text
        )
        query_terms = [
            t for t in query_analysis["tokens"]
            if t in text.lower() and len(t) > 3
        ]
        combined = list(dict.fromkeys(entities[:4] + stats[:2] + query_terms[:3]))
        return combined[:6]


# ─── Knowledge Gap Detector ───────────────────────────────────────────────────

class KnowledgeGapDetector:
    """
    Identifies what the retrieved evidence does NOT cover — critical for
    decision-support to avoid overconfident answers.
    """

    def detect_gaps(
        self,
        query_analysis: dict,
        evidence: list[EvidenceNode],
        all_results: list[RetrievedChunk],
        corpus_chunk_count: int = 0,
    ) -> list[str]:
        """Return list of identified knowledge gaps."""
        gaps = []

        # Coverage check: are all query entities mentioned in evidence?
        covered_text = ' '.join(e.excerpt for e in evidence).lower()
        for entity in query_analysis["entities"]:
            if entity.lower() not in covered_text:
                gaps.append(
                    f"No direct evidence found for '{entity}' in the indexed documents"
                )

        # Quantitative gap
        if query_analysis["needs_quantitative"]:
            has_stats = any(
                re.search(r'\d+\.?\d*%|\bp\s*[<=>]|odds ratio|hazard ratio', e.excerpt.lower())
                for e in evidence
            )
            if not has_stats:
                gaps.append(
                    "Quantitative data (effect sizes, p-values, confidence intervals) "
                    "not found in retrieved evidence"
                )

        # Contradictory evidence gap
        contradictions = [e for e in evidence if e.support_type == "contradictory"]
        if contradictions:
            gaps.append(
                f"Contradictory evidence present in {len(contradictions)} source(s) — "
                "answer may reflect incomplete consensus"
            )

        # Low score gap — threshold scales with corpus size.
        # BM25 IDF for a rare term ≈ log((N + 0.5) / 1.5 + 1); a single
        # well-matched term contributes roughly idf × (K1+1) ≈ idf × 2.5.
        # We flag when the top score falls below 1.5 × idf_rare, meaning
        # not even one term matched strongly — this scales correctly as N grows.
        if all_results:
            n = max(corpus_chunk_count, 1)
            idf_rare = math.log((n + 0.5) / 1.5 + 1)
            low_score_threshold = idf_rare * 1.5
            if all_results[0].score < low_score_threshold:
                gaps.append(
                    "Retrieved evidence has low relevance scores — "
                    "the documents may not directly address this query"
                )

        # Recency gap (heuristic)
        old_docs = [e for e in evidence if re.search(r'\b(19|200[0-5])\d{2}\b', e.excerpt)]
        if len(old_docs) >= len(evidence) // 2 and evidence:
            gaps.append(
                "Evidence may be from older literature — consider verifying with recent publications"
            )

        return gaps[:5]  # cap for readability


# ─── Answer Synthesizer ───────────────────────────────────────────────────────

class AnswerSynthesizer:
    """
    Synthesizes retrieved evidence into a structured answer with an
    explicit reasoning chain — the core of the decision-support system.
    """

    def synthesize(
        self,
        query_analysis: dict,
        evidence: list[EvidenceNode],
        gaps: list[str],
    ) -> tuple[str, list[ReasoningStep], float]:
        """
        Returns (answer_text, reasoning_chain, confidence_score).
        Confidence is derived from evidence quality, not LLM certainty.
        """
        reasoning = []
        step = 1

        # Step 1: Query interpretation
        reasoning.append(ReasoningStep(
            step_number=step,
            label="Query interpretation",
            content=(
                f"Query classified as '{query_analysis['intent']}' intent. "
                f"Key entities identified: {', '.join(query_analysis['entities']) if query_analysis['entities'] else 'none explicitly named'}. "
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

        # Step 2: Evidence assessment
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

        # Step 3: Synthesis
        answer_parts = self._build_answer(query_analysis, evidence)
        reasoning.append(ReasoningStep(
            step_number=step,
            label="Answer synthesis",
            content=(
                f"Synthesized answer from {len(evidence)} source excerpts. "
                f"Primary evidence from: {evidence[0].doc_title} (section: {evidence[0].section})."
                + (f" Note: contradictory evidence in corpus." if contra_count > 0 else "")
            ),
            confidence=self._compute_synthesis_confidence(evidence),
        ))
        step += 1

        # Step 4: Gap acknowledgment
        if gaps:
            reasoning.append(ReasoningStep(
                step_number=step,
                label="Knowledge gap analysis",
                content=f"Identified {len(gaps)} gap(s): {'; '.join(gaps[:2])}.",
                confidence=0.7,
            ))
            step += 1

        # Final confidence
        confidence = self._compute_overall_confidence(evidence, gaps, direct_count)

        return '\n\n'.join(answer_parts), reasoning, confidence

    def _build_answer(self, query_analysis: dict, evidence: list[EvidenceNode]) -> list[str]:
        """Construct a structured answer from evidence."""
        parts = []
        intent = query_analysis["intent"]

        # Opening statement
        direct_evidence = [e for e in evidence if e.support_type == "direct"]
        indirect_evidence = [e for e in evidence if e.support_type == "indirect"]
        contradictory = [e for e in evidence if e.support_type == "contradictory"]

        if direct_evidence:
            parts.append(
                f"Based on direct evidence from {len(direct_evidence)} source(s), "
                f"the following can be stated regarding your query:"
            )
            for i, e in enumerate(direct_evidence[:3], 1):
                excerpt_summary = TextProcessor.truncate_at_sentence(e.excerpt, 300)
                if len(excerpt_summary) < len(e.excerpt):
                    excerpt_summary += "..."
                parts.append(f"[{i}] From '{e.doc_title}' ({e.section}): {excerpt_summary}")
        elif indirect_evidence:
            parts.append(
                "No direct statements were found, but the following indirect evidence is relevant:"
            )
            for i, e in enumerate(indirect_evidence[:3], 1):
                excerpt_summary = TextProcessor.truncate_at_sentence(e.excerpt, 280)
                if len(excerpt_summary) < len(e.excerpt):
                    excerpt_summary += "..."
                parts.append(f"[{i}] From '{e.doc_title}' ({e.section}): {excerpt_summary}")
        else:
            parts.append("Evidence retrieval returned limited relevant content for this specific query.")

        if contradictory:
            parts.append(
                f"⚠ Contradictory evidence found in {len(contradictory)} source(s): "
                + "; ".join(
                    f"'{e.doc_title}' states: {TextProcessor.truncate_at_sentence(e.excerpt, 150)}..."
                    for e in contradictory[:2]
                )
            )

        # Synthesis statement
        if len(evidence) >= 2:
            common_terms = self._find_common_terms(evidence)
            if common_terms:
                parts.append(
                    f"Cross-source analysis: The terms {', '.join(common_terms[:4])} "
                    f"appear consistently across multiple sources, suggesting convergent evidence."
                )

        return parts

    def _find_common_terms(self, evidence: list[EvidenceNode]) -> list[str]:
        """Find terms that appear in multiple evidence nodes."""
        if len(evidence) < 2:
            return []
        term_counts: Counter = Counter()
        for e in evidence:
            term_counts.update(set(e.key_terms))
        return [term for term, count in term_counts.most_common(6) if count >= 2]

    def _compute_synthesis_confidence(self, evidence: list[EvidenceNode]) -> float:
        """Compute confidence based on evidence quality."""
        if not evidence:
            return 0.0
        avg_relevance = sum(e.relevance_score for e in evidence) / len(evidence)
        direct_ratio = sum(1 for e in evidence if e.support_type == "direct") / len(evidence)
        contra_penalty = sum(1 for e in evidence if e.support_type == "contradictory") * 0.1
        return max(0.0, min(1.0, avg_relevance * 0.5 + direct_ratio * 0.5 - contra_penalty))

    def _compute_overall_confidence(
        self,
        evidence: list[EvidenceNode],
        gaps: list[str],
        direct_count: int,
    ) -> float:
        """Final confidence score for the decision output."""
        base = 0.3
        evidence_bonus = min(0.4, len(evidence) * 0.08)
        direct_bonus = min(0.2, direct_count * 0.07)
        gap_penalty = len(gaps) * 0.05
        return max(0.05, min(0.95, base + evidence_bonus + direct_bonus - gap_penalty))


# ─── Follow-up Generator ─────────────────────────────────────────────────────

class FollowUpGenerator:
    """Generates contextually relevant follow-up questions."""

    TEMPLATES = {
        "mechanism": [
            "What are the downstream effects of {entity}?",
            "Which molecular targets are involved in this pathway?",
            "Are there known inhibitors or activators of this mechanism?",
        ],
        "comparison": [
            "What are the contraindications for each option?",
            "How do these options compare in specific patient populations?",
            "What does meta-analytic evidence say about this comparison?",
        ],
        "treatment": [
            "What is the recommended dosage range and administration route?",
            "What are the most common adverse effects reported?",
            "Is there evidence of treatment resistance or tolerance?",
        ],
        "diagnosis": [
            "What is the sensitivity and specificity of this diagnostic approach?",
            "What are the differential diagnoses to consider?",
            "Are there validated clinical scoring systems for this condition?",
        ],
        "general": [
            "Can you provide more detail on the methodology used in these studies?",
            "Are there clinical guidelines that address this topic?",
            "What patient populations were studied?",
        ],
    }

    def generate(self, query_analysis: dict, evidence: list[EvidenceNode]) -> list[str]:
        """Generate 3 relevant follow-up questions."""
        intent = query_analysis["intent"]
        templates = self.TEMPLATES.get(intent, self.TEMPLATES["general"])

        questions = []
        entities = query_analysis["entities"]

        for template in templates[:3]:
            if "{entity}" in template and entities:
                q = template.format(entity=entities[0])
            else:
                q = template.replace("{entity}", "this target")
            questions.append(q)

        # Add a gap-driven question if applicable
        key_terms = []
        for e in evidence[:2]:
            key_terms.extend(e.key_terms[:2])
        if key_terms:
            unique_terms = list(dict.fromkeys(key_terms))[:2]
            questions.append(
                f"What is the clinical significance of {' and '.join(unique_terms)}?"
            )

        return questions[:4]


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

class BioRAGEngine:
    """
    Main decision-support RAG engine.
    Orchestrates the full pipeline: index → retrieve → rerank → classify → synthesize.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        retrieval_top_k: int = 15,
        rerank_top_k: int = 5,
        synthesizer: AnswerSynthesizer | None = None,
    ):
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.index = InvertedIndex()
        self.query_analyzer = QueryAnalyzer()
        self.reranker = Reranker()
        self.evidence_classifier = EvidenceClassifier()
        self.gap_detector = KnowledgeGapDetector()
        self.synthesizer = synthesizer if synthesizer is not None else AnswerSynthesizer()
        self.followup_gen = FollowUpGenerator()

        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

        self.documents: dict[str, dict] = {}  # doc_id -> metadata
        self._lock = threading.Lock()  # serialises concurrent add_document / query

    def add_document(self, doc_id: str, title: str, text: str, metadata: dict = None) -> int:
        """Add a document to the index. Returns number of chunks created."""
        with self._lock:
            chunks = self.chunker.chunk_document(doc_id, title, text)
            for chunk in chunks:
                self.index.add_chunk(chunk)

            self.documents[doc_id] = {
                "id": doc_id,
                "title": title,
                "chunks": len(chunks),
                "metadata": metadata or {},
            }
            return len(chunks)

    def query(self, question: str) -> DecisionOutput:
        """
        Full decision-support pipeline for a user query.
        Returns a structured DecisionOutput with evidence, reasoning, and gaps.
        """
        with self._lock:
            # 1. Analyze query
            q_analysis = self.query_analyzer.analyze(question)

            # 2. Retrieve
            raw_results = self.index.search(
                q_analysis["expanded_tokens"],
                top_k=self.retrieval_top_k,
            )

            # 3. Rerank
            reranked = self.reranker.rerank(raw_results, q_analysis, top_k=self.rerank_top_k)

            # 4. Build evidence nodes
            evidence_nodes: list[EvidenceNode] = []
            for r in reranked:
                support_type, key_terms = self.evidence_classifier.classify(r.chunk, q_analysis)

                # Normalize score to 0-1 range for relevance display
                max_score = reranked[0].score if reranked else 1.0
                relevance = min(1.0, r.score / max(max_score, 0.01))

                node = EvidenceNode(
                    source_id=r.chunk.id,
                    doc_title=r.chunk.doc_title,
                    section=r.chunk.section,
                    excerpt=TextProcessor.truncate_at_sentence(r.chunk.text, 400),
                    relevance_score=round(relevance, 3),
                    support_type=support_type,
                    key_terms=key_terms,
                )
                evidence_nodes.append(node)

            # 5. Detect knowledge gaps
            gaps = self.gap_detector.detect_gaps(
                q_analysis, evidence_nodes, raw_results, self.index.doc_count
            )

            # 6. Synthesize answer + reasoning chain
            answer, reasoning_chain, confidence = self.synthesizer.synthesize(
                q_analysis, evidence_nodes, gaps
            )

            # 7. Generate follow-up questions
            follow_ups = self.followup_gen.generate(q_analysis, evidence_nodes)

            # 8. Confidence label
            if confidence >= 0.75:
                conf_label = "High"
            elif confidence >= 0.50:
                conf_label = "Moderate"
            elif confidence >= 0.25:
                conf_label = "Low"
            else:
                conf_label = "Insufficient"

            return DecisionOutput(
                query=question,
                answer=answer,
                confidence=round(confidence, 3),
                confidence_label=conf_label,
                evidence=evidence_nodes,
                reasoning_chain=reasoning_chain,
                knowledge_gaps=gaps,
                follow_up_questions=follow_ups,
                sources_used=len(set(e.doc_title for e in evidence_nodes)),
                total_chunks_searched=len(self.index.chunks),
            )

    def get_corpus_stats(self) -> dict:
        """Return statistics about the indexed corpus.

        When documents have been ingested via the PubMed/PMC pipeline the
        ``full_text_documents`` and ``abstract_only_documents`` counts reflect
        how many articles have complete body text versus abstract only.
        """
        with self._lock:
            idx_stats = self.index.stats()
            full_text = sum(
                1 for d in self.documents.values()
                if d["metadata"].get("has_full_text", False)
            )
            return {
                "documents": len(self.documents),
                "chunks": idx_stats["total_chunks"],
                "unique_terms": idx_stats["unique_terms"],
                "avg_chunk_length": idx_stats["avg_chunk_length"],
                "full_text_documents": full_text,
                "abstract_only_documents": len(self.documents) - full_text,
                "document_list": [
                    {
                        "id": d["id"],
                        "title": d["title"],
                        "chunks": d["chunks"],
                        "source": d["metadata"].get("source", "local"),
                        "has_full_text": d["metadata"].get("has_full_text", True),
                    }
                    for d in self.documents.values()
                ],
            }

    def to_dict(self, output: DecisionOutput) -> dict:
        """Serialize a DecisionOutput to a plain dict."""
        return asdict(output)

"""Evaluation metrics for paraphrase generation.

A good paraphrase is (a) semantically faithful to the source and (b) lexically
different from it, so we report one metric from each family:

* adequacy:  semantic similarity (sentence embeddings, BoW-cosine fallback)
* diversity: PINC (Chen & Dolan, 2011), self-BLEU against the source,
             Jaccard overlap, and edit-distance ratio
* reference-based (if gold paraphrases exist): BLEU and ROUGE-L
"""
from __future__ import annotations

import csv
import logging
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from statistics import mean
from typing import Dict, List, Optional, Sequence

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_SMOOTH = SmoothingFunction().method1
_BOW_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "with", "as", "by",
}


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------- diversity
def pinc(source: str, candidate: str, max_n: int = 4) -> float:
    """Paraphrase In N-gram Changes: 1 - |ngrams(src) & ngrams(cand)| / |ngrams(cand)|,
    averaged over n = 1..max_n. 0 = copy, 1 = no n-gram shared with the source."""
    s, c = tokenize(source), tokenize(candidate)
    scores = []
    for n in range(1, max_n + 1):
        cand = set(ngrams(c, n))
        if not cand:
            continue
        scores.append(1.0 - len(cand & set(ngrams(s, n))) / len(cand))
    return sum(scores) / len(scores) if scores else 0.0


def self_bleu(source: str, candidate: str) -> float:
    """BLEU of the candidate against the *source* (lower = more different)."""
    s, c = tokenize(source), tokenize(candidate)
    if not s or not c:
        return 0.0
    return float(sentence_bleu([s], c, smoothing_function=_SMOOTH))


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def edit_ratio(a: str, b: str) -> float:
    """Character-level similarity in [0, 1] (1 = identical)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ------------------------------------------------------- reference-based
def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 based on the longest common subsequence of tokens."""
    r, h = tokenize(reference), tokenize(hypothesis)
    if not r or not h:
        return 0.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if r[i - 1] == h[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(h), lcs / len(r)
    return 2 * p * rec / (p + rec)


def bleu_vs_reference(reference: str, hypothesis: str) -> float:
    r, h = tokenize(reference), tokenize(hypothesis)
    if not r or not h:
        return 0.0
    return float(sentence_bleu([r], h, smoothing_function=_SMOOTH))


# ---------------------------------------------------------------- adequacy
class SemanticScorer:
    """Semantic similarity between two texts.

    backend="embedding": sentence-transformers cosine similarity (recommended).
    backend="bow":       bag-of-words cosine (no download; blind to synonyms).
    backend="auto":      embedding if it can be loaded, else bow.
    """

    def __init__(self, backend: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.backend = "bow"
        self._model = None
        if backend in ("auto", "embedding"):
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                self.backend = "embedding"
            except Exception as exc:
                if backend == "embedding":
                    raise
                logger.warning("Sentence-embedding model unavailable (%s); using BoW cosine.", exc)

    @property
    def default_min_similarity(self) -> float:
        return 0.75 if self.backend == "embedding" else 0.3

    def similarity(self, a: str, b: str) -> float:
        if self.backend == "embedding":
            va, vb = self._model.encode([a, b], normalize_embeddings=True)
            return max(0.0, float(sum(x * y for x, y in zip(va, vb))))
        return self._bow_cosine(a, b)

    @staticmethod
    def _bow_cosine(a: str, b: str) -> float:
        ca = Counter(t for t in tokenize(a) if t not in _BOW_STOP)
        cb = Counter(t for t in tokenize(b) if t not in _BOW_STOP)
        dot = sum(ca[t] * cb[t] for t in ca)
        na = math.sqrt(sum(v * v for v in ca.values()))
        nb = math.sqrt(sum(v * v for v in cb.values()))
        return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------- aggregate
def evaluate_pair(source: str, candidate: str, scorer: Optional[SemanticScorer] = None,
                  reference: Optional[str] = None) -> Dict[str, float]:
    scorer = scorer or SemanticScorer("bow")
    out = {
        "semantic_sim": scorer.similarity(source, candidate),
        "pinc": pinc(source, candidate),
        "self_bleu": self_bleu(source, candidate),
        "jaccard": jaccard(source, candidate),
        "edit_ratio": edit_ratio(source, candidate),
    }
    if reference:
        out["bleu_ref"] = bleu_vs_reference(reference, candidate)
        out["rouge_l_ref"] = rouge_l(reference, candidate)
    return out


def evaluate_dataset(sources: List[str], outputs: List[str], references: Optional[List[str]] = None,
                     scorer: Optional[SemanticScorer] = None) -> Dict[str, object]:
    """Per-example metrics plus means (and corpus BLEU when references are given)."""
    if len(sources) != len(outputs):
        raise ValueError("sources and outputs must have the same length")
    scorer = scorer or SemanticScorer("bow")
    rows = []
    for i, (s, o) in enumerate(zip(sources, outputs)):
        ref = references[i] if references and references[i] else None
        rows.append(evaluate_pair(s, o, scorer, ref))
    summary: Dict[str, float] = {}
    for key in rows[0] if rows else []:
        summary[key] = mean(r[key] for r in rows if key in r)
    if references and any(references):
        pairs = [(r, o) for r, o in zip(references, outputs) if r]
        summary["corpus_bleu_ref"] = float(
            corpus_bleu([[tokenize(r)] for r, _ in pairs], [tokenize(o) for _, o in pairs],
                        smoothing_function=_SMOOTH)
        )
    summary["semantic_backend"] = scorer.backend  # type: ignore[assignment]
    return {"per_example": rows, "summary": summary}


def load_csv(path: str) -> Dict[str, List[str]]:
    """Read a CSV with a `source` column and an optional `reference` column."""
    sources, refs = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source", "").strip():
                sources.append(row["source"].strip())
                refs.append((row.get("reference") or "").strip())
    return {"sources": sources, "references": refs}

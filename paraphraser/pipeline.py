"""High-level engine: generate candidates, score them, rerank, return the best."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .evaluate import SemanticScorer, pinc
from .utils import dedupe_keep_order

METHODS = ("rule", "neural", "hybrid")


@dataclass
class Candidate:
    text: str
    method: str
    similarity: float = 0.0
    pinc: float = 0.0
    score: float = 0.0
    accepted: bool = True
    extra: Dict[str, float] = field(default_factory=dict)


class ParaphraseEngine:
    """Combines the rule-based and neural paraphrasers with a reranker.

    Reranking score = alpha * semantic_similarity + (1 - alpha) * PINC, and a
    candidate is only *accepted* if similarity >= min_similarity. This pushes
    the output towards "same meaning, different words".
    """

    def __init__(self, rule=None, neural=None, scorer: Optional[SemanticScorer] = None,
                 neural_model: Optional[str] = None, seed: Optional[int] = None,
                 scorer_backend: str = "auto"):
        self._rule = rule
        self._neural = neural
        self._neural_model = neural_model
        self._scorer = scorer
        self._scorer_backend = scorer_backend
        self._seed = seed

    # lazily-created components -------------------------------------------------
    @property
    def rule(self):
        if self._rule is None:
            from .rule_based import RuleBasedParaphraser

            self._rule = RuleBasedParaphraser(seed=self._seed)
        return self._rule

    @property
    def neural(self):
        if self._neural is None:
            from .neural import DEFAULT_MODEL, NeuralParaphraser

            self._neural = NeuralParaphraser(self._neural_model or DEFAULT_MODEL)
        return self._neural

    @property
    def scorer(self) -> SemanticScorer:
        if self._scorer is None:
            self._scorer = SemanticScorer(self._scorer_backend)
        return self._scorer

    # ---------------------------------------------------------------------------
    def paraphrase(self, text: str, method: str = "hybrid", n: int = 3, alpha: float = 0.6,
                   min_similarity: Optional[float] = None, **gen_kwargs) -> List[Candidate]:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        text = text.strip()
        if not text:
            return []

        pool: List[tuple] = []  # (text, method)
        if method in ("neural", "hybrid"):
            pool += [(t, "neural") for t in self.neural.paraphrase(text, n=n, **gen_kwargs)]
        if method in ("rule", "hybrid"):
            pool += [(t, "rule") for t in self.rule.paraphrase(text, n=n)]

        seen = set()
        unique = []
        for t, m in pool:
            key = t.strip().lower()
            if key and key != text.lower() and key not in seen:
                seen.add(key)
                unique.append((t, m))

        threshold = self.scorer.default_min_similarity if min_similarity is None else min_similarity
        cands = []
        for t, m in unique:
            sim = self.scorer.similarity(text, t)
            div = pinc(text, t)
            cands.append(Candidate(t, m, sim, div, alpha * sim + (1 - alpha) * div, sim >= threshold))
        cands.sort(key=lambda c: (c.accepted, c.score), reverse=True)
        return cands[:n]

"""Automatic text paraphraser (rule-based + neural) with evaluation tools."""
from .evaluate import SemanticScorer, evaluate_dataset, evaluate_pair, pinc
from .pipeline import Candidate, ParaphraseEngine

__all__ = [
    "ParaphraseEngine",
    "Candidate",
    "SemanticScorer",
    "evaluate_pair",
    "evaluate_dataset",
    "pinc",
]
__version__ = "1.0.0"

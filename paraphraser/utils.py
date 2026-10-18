"""Shared helpers: NLTK resource bootstrap, sentence splitting, casing utilities."""
from __future__ import annotations

import logging
import re
from typing import List

import nltk

logger = logging.getLogger(__name__)

# package name -> path used by nltk.data.find
_NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "stopwords": "corpora/stopwords",
}
_ready = False


def _has_resource(path: str) -> bool:
    for candidate in (path, path + ".zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_resources(quiet: bool = True) -> bool:
    """Download any missing NLTK data. Returns True if everything is available."""
    global _ready
    if _ready:
        return True
    ok = True
    for pkg, path in _NLTK_RESOURCES.items():
        if _has_resource(path):
            continue
        try:
            nltk.download(pkg, quiet=quiet)
        except Exception as exc:  # network problems etc.
            logger.warning("Could not download NLTK resource %s: %s", pkg, exc)
        if not _has_resource(path):
            ok = False
    _ready = ok
    return ok


_FALLBACK_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences (NLTK punkt, with a regex fallback)."""
    text = text.strip()
    if not text:
        return []
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        return _FALLBACK_SPLIT.split(text)


def match_case(original: str, new: str) -> str:
    """Give `new` the capitalisation pattern of `original`."""
    if len(original) > 1 and original.isupper():
        return new.upper()
    if original[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


_AN_EXCEPTIONS_A = ("uni", "use", "usu", "eu", "one", "once", "ubi")
_A_EXCEPTIONS_AN = ("hour", "honest", "honor", "honour", "heir")


def indefinite_article(next_word: str) -> str:
    """Return 'a' or 'an' for the given following word (simple heuristic)."""
    w = next_word.lower()
    if w.startswith(_A_EXCEPTIONS_AN):
        return "an"
    if w.startswith(_AN_EXCEPTIONS_A):
        return "a"
    return "an" if w[:1] in "aeiou" else "a"


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out

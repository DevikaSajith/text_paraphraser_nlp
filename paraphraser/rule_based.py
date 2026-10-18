"""Rule-based paraphraser.

Pipeline for every sentence
---------------------------
1. Syntactic rewrites (regex level):
   * discourse-connective swaps       ("However," -> "Nevertheless,")
   * contraction expansion/contraction ("don't" <-> "do not")
   * subordinate-clause reordering    ("Because X, Y." <-> "Y because X.")
2. Lexical substitution:
   * tokenise + POS-tag (NLTK)
   * disambiguate the word sense with the Lesk algorithm
   * pick a WordNet synonym from the chosen synset
   * re-inflect it to the original POS tag (lemminflect, optional)
   * preserve capitalisation and fix a/an agreement

It needs no GPU and no model download, so it works as a fast baseline.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.wsd import lesk

from .utils import (
    dedupe_keep_order,
    ensure_nltk_resources,
    indefinite_article,
    match_case,
    split_sentences,
)

try:  # optional, gives proper inflection of replacement words
    from lemminflect import getInflection, getLemma

    HAS_LEMMINFLECT = True
except ImportError:  # pragma: no cover
    HAS_LEMMINFLECT = False

_PTB_TO_WN = {"NN": "n", "VB": "v", "JJ": "a", "RB": "r"}
_WN_TO_UPOS = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}
_BASE_TAGS = {"NN", "VB", "VBP", "JJ", "RB"}  # usable without an inflection library

# very polysemous / function-like words that give poor substitutions
_BLOCKLIST = {
    "be", "have", "do", "say", "get", "make", "go", "take", "put", "let",
    "give", "come", "see", "know", "thing", "way", "also", "very", "just",
    "not", "never", "really", "much", "many", "more", "most", "other",
}

_CONNECTIVES: Dict[str, str] = {
    "However": "Nevertheless",
    "Nevertheless": "However",
    "Moreover": "Furthermore",
    "Furthermore": "Moreover",
    "Therefore": "Consequently",
    "Consequently": "Therefore",
    "Thus": "Hence",
    "Hence": "Thus",
    "Additionally": "In addition",
    "In addition": "Additionally",
    "For example": "For instance",
    "For instance": "For example",
    "Finally": "Lastly",
    "Overall": "On the whole",
    "In conclusion": "To conclude",
    "Meanwhile": "In the meantime",
}
_CONNECTIVE_RE = re.compile(
    r"^(?P<c>" + "|".join(sorted(map(re.escape, _CONNECTIVES), key=len, reverse=True)) + r")(?P<rest>,.*)$",
    re.DOTALL,
)

_EXPAND = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "wouldn't": "would not",
    "shouldn't": "should not", "couldn't": "could not",
    "i'm": "I am", "they're": "they are", "we're": "we are", "you're": "you are",
    "i've": "I have", "we've": "we have", "they've": "they have",
}
_CONTRACT = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "cannot": "can't", "will not": "won't", "is not": "isn't",
    "are not": "aren't", "was not": "wasn't", "were not": "weren't",
    "has not": "hasn't", "have not": "haven't", "would not": "wouldn't",
    "should not": "shouldn't", "could not": "couldn't",
}
_EXPAND_RE = re.compile(r"\b(" + "|".join(map(re.escape, _EXPAND)) + r")\b", re.IGNORECASE)
_CONTRACT_RE = re.compile(r"\b(" + "|".join(map(re.escape, _CONTRACT)) + r")\b", re.IGNORECASE)

_FRONT_RE = re.compile(r"^(Because|Although|Though|If|Unless)\s+(.+?),\s+(.+?)([.!?])$", re.DOTALL)
_BACK_RE = re.compile(r"^(.+?)\s+(because|although|though|unless)\s+(.+?)([.!?])$", re.DOTALL)


class RuleBasedParaphraser:
    """Synonym-substitution + light syntactic rewriting paraphraser."""

    def __init__(
        self,
        replace_prob: float = 0.6,
        max_synonyms: int = 5,
        use_syntax: bool = True,
        seed: Optional[int] = None,
    ):
        if not ensure_nltk_resources():
            raise RuntimeError(
                "NLTK resources (punkt_tab, tagger, wordnet, omw-1.4, stopwords) are missing and "
                "could not be downloaded. Run: python -m nltk.downloader punkt_tab "
                "averaged_perceptron_tagger_eng wordnet omw-1.4 stopwords"
            )
        from nltk.corpus import stopwords

        self.stop = set(stopwords.words("english"))
        self.replace_prob = replace_prob
        self.max_synonyms = max_synonyms
        self.use_syntax = use_syntax
        self.rng = random.Random(seed)
        self._detok = TreebankWordDetokenizer()

    # ------------------------------------------------------------------ public
    def paraphrase(self, text: str, n: int = 1, **_unused) -> List[str]:
        """Return up to `n` distinct paraphrases of `text`."""
        sentences = split_sentences(text)
        if not sentences:
            return []
        source_key = text.strip().lower()
        outputs: List[str] = []
        attempts = 0
        while len(outputs) < n and attempts < n * 6:
            attempts += 1
            candidate = " ".join(self._paraphrase_sentence(s) for s in sentences)
            if candidate.strip().lower() != source_key:
                outputs.append(candidate)
            outputs = dedupe_keep_order(outputs)
        return outputs[:n]

    # --------------------------------------------------------------- syntactic
    def _swap_connective(self, sentence: str) -> str:
        m = _CONNECTIVE_RE.match(sentence)
        if m and self.rng.random() < 0.8:
            return _CONNECTIVES[m.group("c")] + m.group("rest")
        return sentence

    def _toggle_contractions(self, sentence: str) -> str:
        def keep_case(match: re.Match, table: Dict[str, str]) -> str:
            word = match.group(0)
            new = table[word.lower()]
            return new[:1].upper() + new[1:] if word[:1].isupper() else new

        if _EXPAND_RE.search(sentence):
            return _EXPAND_RE.sub(lambda m: keep_case(m, _EXPAND), sentence)
        if self.rng.random() < 0.5:
            return _CONTRACT_RE.sub(lambda m: keep_case(m, _CONTRACT), sentence)
        return sentence

    def _first_word_is_proper(self, clause: str) -> bool:
        tokens = word_tokenize(clause)
        if not tokens:
            return False
        if tokens[0] == "I":
            return True
        tag = nltk.pos_tag(tokens)[0][1]
        return tag in {"NNP", "NNPS"}

    def _reorder_clauses(self, sentence: str) -> str:
        if self.rng.random() >= 0.5:
            return sentence
        m = _FRONT_RE.match(sentence)
        if m:  # "Because X, Y."  ->  "Y because X."
            conj, sub, main, punct = m.groups()
            main = main[:1].upper() + main[1:]
            return f"{main} {conj.lower()} {sub}{punct}"
        m = _BACK_RE.match(sentence)
        if m and "," not in m.group(1):  # "Y because X."  ->  "Because X, y."
            main, conj, sub, punct = m.groups()
            if not self._first_word_is_proper(main):
                main = main[:1].lower() + main[1:]
            return f"{conj.capitalize()} {sub}, {main}{punct}"
        return sentence

    def _syntactic_transform(self, sentence: str) -> str:
        sentence = self._swap_connective(sentence)
        sentence = self._toggle_contractions(sentence)
        sentence = self._reorder_clauses(sentence)
        return sentence

    # ----------------------------------------------------------------- lexical
    def _lemma_of(self, token: str, wn_pos: str) -> str:
        if HAS_LEMMINFLECT:
            lemmas = getLemma(token.lower(), upos=_WN_TO_UPOS[wn_pos])
            if lemmas:
                return lemmas[0]
        return token.lower()

    def _inflect(self, lemma: str, tag: str) -> str:
        if HAS_LEMMINFLECT:
            forms = getInflection(lemma, tag=tag)
            if forms:
                return forms[0]
        return lemma

    def _synonyms(self, lemma: str, synset) -> List[str]:
        antonyms = {
            a.name()
            for l in synset.lemmas()
            if l.name().lower() == lemma
            for a in l.antonyms()
        }
        found: List[str] = []
        for l in sorted(synset.lemmas(), key=lambda x: -x.count()):
            name = l.name()
            if not name.isalpha() or name != name.lower():  # no phrases, hyphens, proper nouns
                continue
            if name == lemma or name in antonyms or name in found:
                continue
            found.append(name)
        return found[: self.max_synonyms]

    def _replacement(self, token: str, tag: str, context: List[str]) -> Optional[str]:
        wn_pos = _PTB_TO_WN.get(tag[:2])
        if wn_pos is None or tag in {"NNP", "NNPS"}:
            return None
        if len(token) < 4 or not token.isalpha() or token.lower() in self.stop:
            return None
        if not HAS_LEMMINFLECT and tag not in _BASE_TAGS:
            return None
        lemma = self._lemma_of(token, wn_pos)
        if lemma in _BLOCKLIST:
            return None
        sense = lesk(context, lemma, pos=wn_pos)
        if sense is None:
            return None
        options = self._synonyms(lemma, sense)
        if not options:
            return None
        choice = self.rng.choice(options)
        return match_case(token, self._inflect(choice, tag))

    def _paraphrase_sentence(self, sentence: str) -> str:
        if self.use_syntax:
            sentence = self._syntactic_transform(sentence)
        tokens = word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        out = list(tokens)
        changed = set()
        for i, (tok, tag) in enumerate(tagged):
            if self.rng.random() > self.replace_prob:
                continue
            new = self._replacement(tok, tag, tokens)
            if new and new.lower() != tok.lower():
                out[i] = new
                changed.add(i)
        # a / an agreement for words we changed
        for i in changed:
            if i > 0 and out[i - 1].lower() in {"a", "an"}:
                art = indefinite_article(out[i])
                out[i - 1] = match_case(out[i - 1], art)
        return self._detok.detokenize(out)

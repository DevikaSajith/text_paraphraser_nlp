import pytest

from paraphraser import ParaphraseEngine, SemanticScorer


class FakeNeural:
    """Stands in for the T5 model so tests need no download."""

    def paraphrase(self, text, n=3, **kw):
        return [
            "the cat sat on the mat",             # identical to source -> dropped
            "a cat was sitting on the mat",       # similar and different wording
            "stock markets rallied on tuesday",   # unrelated -> low similarity
        ][:n]


class FakeRule:
    def paraphrase(self, text, n=1, **kw):
        return ["the cat sat upon the mat"]


def make_engine():
    return ParaphraseEngine(rule=FakeRule(), neural=FakeNeural(), scorer=SemanticScorer("bow"))


def test_invalid_method():
    with pytest.raises(ValueError):
        make_engine().paraphrase("x", method="magic")


def test_empty_text():
    assert make_engine().paraphrase("  ") == []


def test_identical_candidates_removed_and_unrelated_flagged():
    res = make_engine().paraphrase("the cat sat on the mat", method="hybrid", n=5)
    texts = [c.text for c in res]
    assert "the cat sat on the mat" not in texts
    unrelated = next(c for c in res if "stock" in c.text)
    assert unrelated.accepted is False
    assert res[-1] is unrelated  # rejected candidates rank last


def test_methods_select_sources():
    e = make_engine()
    assert {c.method for c in e.paraphrase("the cat sat on the mat", method="rule", n=3)} == {"rule"}
    assert {c.method for c in e.paraphrase("the cat sat on the mat", method="neural", n=3)} == {"neural"}


def test_n_limits_results():
    assert len(make_engine().paraphrase("the cat sat on the mat", method="hybrid", n=1)) == 1

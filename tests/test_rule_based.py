import pytest

from paraphraser.utils import ensure_nltk_resources, indefinite_article, match_case, split_sentences

needs_nltk = pytest.mark.skipif(not ensure_nltk_resources(), reason="NLTK data not available")


def test_match_case():
    assert match_case("Big", "large") == "Large"
    assert match_case("BIG", "large") == "LARGE"
    assert match_case("big", "large") == "large"


def test_indefinite_article():
    assert indefinite_article("apple") == "an"
    assert indefinite_article("car") == "a"
    assert indefinite_article("hour") == "an"
    assert indefinite_article("university") == "a"


def test_split_sentences():
    assert len(split_sentences("It rained. We left early.")) == 2


@needs_nltk
def test_rule_based_changes_text_and_is_seeded():
    from paraphraser.rule_based import RuleBasedParaphraser

    text = "The scientist carefully examined the strange samples in the laboratory."
    a = RuleBasedParaphraser(seed=1, replace_prob=1.0).paraphrase(text, n=1)
    b = RuleBasedParaphraser(seed=1, replace_prob=1.0).paraphrase(text, n=1)
    assert a == b  # deterministic given a seed
    assert a and a[0].lower() != text.lower()


@needs_nltk
def test_returns_distinct_candidates():
    from paraphraser.rule_based import RuleBasedParaphraser

    outs = RuleBasedParaphraser(seed=3, replace_prob=1.0).paraphrase(
        "The company announced a significant increase in quarterly profits.", n=3)
    assert len(outs) == len({o.lower() for o in outs})


@needs_nltk
def test_connective_and_contraction_rewrites():
    from paraphraser.rule_based import RuleBasedParaphraser

    p = RuleBasedParaphraser(seed=0, replace_prob=0.0)
    out = p.paraphrase("However, we don't know the answer.", n=1)[0]
    assert out.startswith("Nevertheless,")
    assert "do not" in out


@needs_nltk
def test_clause_reordering_keeps_punctuation_and_content():
    from paraphraser.rule_based import RuleBasedParaphraser

    p = RuleBasedParaphraser(seed=0, replace_prob=0.0)
    results = {p._reorder_clauses("Because it rained, we stayed home.") for _ in range(20)}
    assert "We stayed home because it rained." in results


@needs_nltk
def test_empty_input():
    from paraphraser.rule_based import RuleBasedParaphraser

    assert RuleBasedParaphraser(seed=0).paraphrase("   ") == []

import pytest

from paraphraser.evaluate import (
    SemanticScorer,
    edit_ratio,
    evaluate_dataset,
    evaluate_pair,
    jaccard,
    pinc,
    rouge_l,
    self_bleu,
)

SRC = "the cat sat on the mat"


def test_pinc_identical_is_zero():
    assert pinc(SRC, SRC) == 0.0


def test_pinc_disjoint_is_one():
    assert pinc(SRC, "a dog lay under that rug now") == 1.0


def test_pinc_partial_between_0_and_1():
    assert 0.0 < pinc(SRC, "the cat slept on the sofa") < 1.0


def test_self_bleu_identical_high_and_different_low():
    assert self_bleu(SRC, SRC) > 0.99
    assert self_bleu(SRC, "a dog lay under that rug now") < 0.1


def test_jaccard_and_edit_ratio():
    assert jaccard(SRC, SRC) == 1.0
    assert edit_ratio(SRC, SRC) == 1.0
    assert jaccard("a b c", "d e f") == 0.0


def test_rouge_l():
    assert rouge_l(SRC, SRC) == pytest.approx(1.0)
    assert rouge_l(SRC, "xyz qrs") == 0.0


def test_bow_similarity_ordering():
    s = SemanticScorer("bow")
    assert s.similarity(SRC, SRC) == pytest.approx(1.0)
    assert s.similarity(SRC, "cat on mat") > s.similarity(SRC, "stock markets rallied")


def test_evaluate_pair_keys():
    out = evaluate_pair(SRC, "the cat slept on the sofa", reference="a cat was on the mat")
    assert {"semantic_sim", "pinc", "self_bleu", "jaccard", "edit_ratio", "bleu_ref", "rouge_l_ref"} <= set(out)


def test_evaluate_dataset_summary():
    rep = evaluate_dataset([SRC, "i like tea"], ["the cat slept on the sofa", "i enjoy tea"],
                           references=["a cat was on the mat", "i love tea"])
    assert len(rep["per_example"]) == 2
    assert "corpus_bleu_ref" in rep["summary"]
    assert rep["summary"]["semantic_backend"] == "bow"


def test_evaluate_dataset_length_mismatch():
    with pytest.raises(ValueError):
        evaluate_dataset(["a"], [])

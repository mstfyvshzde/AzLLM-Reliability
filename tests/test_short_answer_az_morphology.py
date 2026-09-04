"""Azerbaijani-aware short-answer matching regression tests."""

from src.evaluation.short_answer_match import (
    short_answer_match_score,
)


def test_matches_azerbaijani_copula_suffix() -> None:
    assert short_answer_match_score(
        "Misirdən keçən çay Nildir.",
        "Nil",
    ) == 1


def test_matches_azerbaijani_case_suffix() -> None:
    assert short_answer_match_score(
        "Minanın daha çox qələmi var.",
        "Mina",
    ) == 1


def test_matches_azerbaijani_relational_form() -> None:
    assert short_answer_match_score(
        "Sahara Afrikan qitəsində yerləşir.",
        "Afrika",
    ) == 1


def test_matches_apostrophe_copula_form() -> None:
    assert short_answer_match_score(
        "Fransa'nın paytaxtı Paris'tir.",
        "Paris",
    ) == 1


def test_matches_hyphenated_numeric_answer() -> None:
    assert short_answer_match_score(
        "200-ün 25 faizi 50-dir.",
        "50",
    ) == 1


def test_matches_hyphenated_fraction_answer() -> None:
    assert short_answer_match_score(
        "Ehtimal 1/2-dir.",
        "1/2",
    ) == 1


def test_does_not_use_arbitrary_prefix_matching() -> None:
    assert short_answer_match_score(
        "Nobody knows the answer.",
        "No",
    ) == 0


def test_does_not_accept_unrelated_answer() -> None:
    assert short_answer_match_score(
        "Arifin daha çox qələmi var.",
        "Mina",
    ) == 0

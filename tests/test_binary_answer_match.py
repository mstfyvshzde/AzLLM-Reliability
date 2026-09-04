"""Binary answer evaluator testleri."""

import pytest

from src.evaluation.binary_answer_match import (
    binary_answer_match_score,
    detect_explicit_binary_polarity,
    is_binary_reference,
    normalize_binary_reference,
)
from src.evaluation.run_inference import PredictionRecord


def make_record(
    prediction: str,
    reference_answer: str,
    item_id: str = "reasoning_001_en",
) -> PredictionRecord:
    return PredictionRecord(
        item_id=item_id,
        pair_id="reasoning_001",
        language="en",
        task="reasoning",
        question="Did Q occur?",
        reference_answer=reference_answer,
        prediction=prediction,
        metadata={
            "category": "logical_reasoning",
            "difficulty": "easy",
            "is_answerable": True,
        },
    )


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("Yes", "positive"),
        ("Bəli", "positive"),
        ("No", "negative"),
        ("Xeyr", "negative"),
    ],
)
def test_normalize_binary_reference(
    reference: str,
    expected: str,
) -> None:
    assert normalize_binary_reference(
        reference
    ) == expected


def test_non_binary_reference_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Reference answer is not binary",
    ):
        normalize_binary_reference(
            "Paris"
        )


def test_detect_explicit_yes() -> None:
    assert detect_explicit_binary_polarity(
        "Yes, Q occurred."
    ) == "positive"


def test_detect_explicit_beli() -> None:
    assert detect_explicit_binary_polarity(
        "Bəli, Q baş verdi."
    ) == "positive"


def test_detect_explicit_no() -> None:
    assert detect_explicit_binary_polarity(
        "No, X cannot be metallic."
    ) == "negative"


def test_detect_explicit_xeyr() -> None:
    assert detect_explicit_binary_polarity(
        "Xeyr, mümkün deyil."
    ) == "negative"


def test_no_explicit_binary_token_returns_none() -> None:
    assert detect_explicit_binary_polarity(
        "Q must have occurred."
    ) is None


def test_matching_explicit_binary_answer() -> None:
    record = make_record(
        prediction="Yes, Q occurred.",
        reference_answer="Yes",
    )

    assert binary_answer_match_score(
        record
    ) == 1


def test_opposite_explicit_binary_answer() -> None:
    record = make_record(
        prediction="Yes, it can.",
        reference_answer="No",
    )

    assert binary_answer_match_score(
        record
    ) == 0


def test_indirect_answer_uses_adjudication() -> None:
    record = make_record(
        prediction="Q must have occurred.",
        reference_answer="Yes",
    )

    assert binary_answer_match_score(
        record,
        adjudication_decisions={
            record.item_id: 1,
        },
    ) == 1


def test_indirect_answer_requires_adjudication() -> None:
    record = make_record(
        prediction="Q must have occurred.",
        reference_answer="Yes",
    )

    with pytest.raises(
        ValueError,
        match="Binary adjudication decisions are required",
    ):
        binary_answer_match_score(
            record
        )


def test_missing_binary_adjudication_item_is_rejected() -> None:
    record = make_record(
        prediction="Q must have occurred.",
        reference_answer="Yes",
    )

    with pytest.raises(
        ValueError,
        match="Missing binary adjudication decision",
    ):
        binary_answer_match_score(
            record,
            adjudication_decisions={},
        )


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("Yes", True),
        ("No", True),
        ("Bəli", True),
        ("Xeyr", True),
        ("Paris", False),
    ],
)
def test_is_binary_reference(
    reference: str,
    expected: bool,
) -> None:
    assert is_binary_reference(
        reference
    ) is expected

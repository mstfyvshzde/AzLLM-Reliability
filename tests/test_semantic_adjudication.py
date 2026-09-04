"""Semantic adjudication altyapısı testleri."""

import pytest

from src.evaluation.run_inference import PredictionRecord
from src.evaluation.semantic_adjudication import (
    create_semantic_adjudication_result,
    exact_semantic_match,
    is_semantic_adjudication_record,
    validate_adjudication_decision,
)


def make_record(
    category: str = "contextual_meaning",
) -> PredictionRecord:
    return PredictionRecord(
        item_id="linguistic_understanding_001_en",
        pair_id="linguistic_understanding_001",
        language="en",
        task="linguistic_understanding",
        question="What does the expression mean?",
        reference_answer="Explain in greater detail",
        prediction="It means explain in greater detail.",
        metadata={
            "category": category,
            "difficulty": "medium",
            "is_answerable": True,
        },
    )


def test_semantic_category_requires_adjudication() -> None:
    assert is_semantic_adjudication_record(
        make_record()
    ) is True


def test_non_semantic_category_does_not_require_adjudication() -> None:
    assert is_semantic_adjudication_record(
        make_record(
            category="reference_resolution"
        )
    ) is False


def test_exact_semantic_match() -> None:
    assert exact_semantic_match(
        "Explain in greater detail",
        "Explain in greater detail",
    ) is True


def test_exact_semantic_match_uses_normalization() -> None:
    assert exact_semantic_match(
        "Explain in greater detail.",
        "Explain in greater detail",
    ) is True


@pytest.mark.parametrize(
    "decision",
    [0, 1],
)
def test_valid_decisions(
    decision: int,
) -> None:
    validate_adjudication_decision(decision)


def test_invalid_decision_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must be 0 or 1",
    ):
        validate_adjudication_decision(2)


def test_create_adjudication_result() -> None:
    record = make_record()

    result = create_semantic_adjudication_result(
        record=record,
        decision=1,
        reason=(
            "Prediction preserves the core meaning "
            "of the reference answer."
        ),
    )

    assert result.item_id == record.item_id
    assert result.category == "contextual_meaning"
    assert result.decision == 1


def test_empty_reason_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="reason cannot be empty",
    ):
        create_semantic_adjudication_result(
            record=make_record(),
            decision=1,
            reason="   ",
        )


def test_load_semantic_adjudication_decisions(
    tmp_path,
) -> None:
    from src.evaluation.semantic_adjudication import (
        load_semantic_adjudication_decisions,
    )

    path = tmp_path / "labels.jsonl"

    path.write_text(
        (
            '{"item_id":"item_1","decision":1,"reason":"Correct."}\n'
            '{"item_id":"item_2","decision":0,"reason":"Incorrect."}\n'
        ),
        encoding="utf-8",
    )

    decisions = load_semantic_adjudication_decisions(
        str(path)
    )

    assert decisions == {
        "item_1": 1,
        "item_2": 0,
    }


def test_duplicate_adjudication_item_is_rejected(
    tmp_path,
) -> None:
    from src.evaluation.semantic_adjudication import (
        load_semantic_adjudication_decisions,
    )

    path = tmp_path / "labels.jsonl"

    path.write_text(
        (
            '{"item_id":"item_1","decision":1}\n'
            '{"item_id":"item_1","decision":0}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate semantic adjudication item_id",
    ):
        load_semantic_adjudication_decisions(
            str(path)
        )

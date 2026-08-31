"""Semantic-answer evaluator testleri."""

from src.evaluation.semantic_answer_match import (
    semantic_answer_match_score,
)


def test_exact_semantic_match() -> None:
    assert semantic_answer_match_score(
        "Came to or achieved",
        "Came to or achieved",
    ) == 1


def test_english_semantic_alias_match() -> None:
    assert semantic_answer_match_score(
        '"Reached" means to come to a conclusion or solution.',
        "Came to or achieved",
    ) == 1


def test_azerbaijani_semantic_alias_match() -> None:
    assert semantic_answer_match_score(
        "Onlar sonda razılığa gəldilər.",
        "Razılıq əldə etdilər",
    ) == 1


def test_unrelated_answer_is_incorrect() -> None:
    assert semantic_answer_match_score(
        "The sentence means they became angry.",
        "Came to or achieved",
    ) == 0


def test_unknown_reference_without_alias_is_incorrect() -> None:
    assert semantic_answer_match_score(
        "Some extended answer.",
        "Unknown reference",
    ) == 0

from src.evaluation.run_inference import PredictionRecord
from src.evaluation.semantic_answer_match import (
    evaluate_semantic_answer_prediction,
)


def make_semantic_record(
    prediction: str,
    reference_answer: str,
) -> PredictionRecord:
    return PredictionRecord(
        item_id="linguistic_001_en",
        pair_id="linguistic_001",
        language="en",
        task="linguistic_understanding",
        question="What does 'reached' mean here?",
        reference_answer=reference_answer,
        prediction=prediction,
        metadata={
            "category": "contextual_meaning",
            "difficulty": "medium",
            "is_answerable": True,
        },
    )


def test_evaluate_semantic_answer_prediction_correct() -> None:
    record = make_semantic_record(
        prediction='"Reached" means to come to a conclusion.',
        reference_answer="Came to or achieved",
    )

    result = evaluate_semantic_answer_prediction(
        record
    )

    assert result.item_id == "linguistic_001_en"
    assert result.semantic_match == 1


def test_evaluate_semantic_answer_prediction_incorrect() -> None:
    record = make_semantic_record(
        prediction="It means they became angry.",
        reference_answer="Came to or achieved",
    )

    result = evaluate_semantic_answer_prediction(
        record
    )

    assert result.semantic_match == 0

from src.evaluation.semantic_answer_match import (
    evaluate_semantic_answer_matches,
    filter_semantic_answer_records,
    is_semantic_answer_task,
)


def test_is_semantic_answer_task() -> None:
    assert is_semantic_answer_task(
        "linguistic_understanding"
    ) is True

    assert is_semantic_answer_task(
        "reasoning"
    ) is False

    assert is_semantic_answer_task(
        "instruction_following"
    ) is False


def test_filter_semantic_answer_records() -> None:
    linguistic_record = make_semantic_record(
        prediction="Reached means to come to a conclusion.",
        reference_answer="Came to or achieved",
    )

    reasoning_record = PredictionRecord(
        item_id="reasoning_001_en",
        pair_id="reasoning_001",
        language="en",
        task="reasoning",
        question="Who is shortest?",
        reference_answer="Nigar",
        prediction="Nigar",
        metadata={
            "category": "comparative_reasoning",
            "difficulty": "easy",
            "is_answerable": True,
        },
    )

    filtered = filter_semantic_answer_records(
        [
            linguistic_record,
            reasoning_record,
        ]
    )

    assert len(filtered) == 1
    assert filtered[0].task == "linguistic_understanding"


def test_evaluate_semantic_answer_matches() -> None:
    records = [
        make_semantic_record(
            prediction="Reached means to come to a conclusion.",
            reference_answer="Came to or achieved",
        )
    ]

    results = evaluate_semantic_answer_matches(
        records
    )

    assert len(results) == 1
    assert results[0].semantic_match == 1


from src.evaluation.semantic_answer_match import (
    calculate_semantic_answer_accuracy,
    summarize_semantic_answer_matches,
)


def test_calculate_semantic_answer_accuracy() -> None:
    records = [
        make_semantic_record(
            prediction="Reached means to come to a conclusion.",
            reference_answer="Came to or achieved",
        ),
        make_semantic_record(
            prediction="It means they became angry.",
            reference_answer="Came to or achieved",
        ),
    ]

    results = evaluate_semantic_answer_matches(
        records
    )

    assert calculate_semantic_answer_accuracy(
        results
    ) == 0.5


def test_summarize_semantic_answer_matches() -> None:
    records = [
        make_semantic_record(
            prediction="Reached means to come to a conclusion.",
            reference_answer="Came to or achieved",
        ),
        make_semantic_record(
            prediction="It means they became angry.",
            reference_answer="Came to or achieved",
        ),
    ]

    results = evaluate_semantic_answer_matches(
        records
    )

    summary = summarize_semantic_answer_matches(
        results
    )

    assert summary == {
        "total": 2,
        "correct": 1,
        "incorrect": 1,
        "accuracy": 0.5,
    }

def test_semantic_alias_with_contradiction_is_incorrect() -> None:
    assert semantic_answer_match_score(
        (
            'Bu cümlədə "razılığa gəldilər" '
            'ifadəsi "became angry" deməkdir.'
        ),
        "Razılıq əldə etdilər",
    ) == 0
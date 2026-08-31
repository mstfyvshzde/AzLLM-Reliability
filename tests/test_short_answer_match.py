"""Short-answer evaluator testleri."""

from src.evaluation.short_answer_match import (
    short_answer_match_score,
)


def test_exact_match_is_correct() -> None:
    assert short_answer_match_score(
        "Nigar",
        "Nigar",
    ) == 1


def test_reference_inside_natural_answer_is_correct() -> None:
    assert short_answer_match_score(
        "Nigar is the shortest.",
        "Nigar",
    ) == 1


def test_wrong_answer_is_incorrect() -> None:
    assert short_answer_match_score(
        "Aysel is the shortest.",
        "Nigar",
    ) == 0


def test_contradictory_answer_is_incorrect() -> None:
    assert short_answer_match_score(
        "Leyla, but actually Nigar finished the presentation.",
        "Leyla",
    ) == 0


def test_case_and_punctuation_are_normalized() -> None:
    assert short_answer_match_score(
        "NIGAR.",
        "Nigar",
    ) == 1


def test_empty_reference_is_incorrect() -> None:
    assert short_answer_match_score(
        "Nigar",
        "",
    ) == 0


from src.evaluation.run_inference import PredictionRecord
from src.evaluation.short_answer_match import (
    evaluate_short_answer_prediction,
)


def make_record(
    prediction: str,
    reference_answer: str,
) -> PredictionRecord:
    return PredictionRecord(
        item_id="reasoning_001_en",
        pair_id="reasoning_001",
        language="en",
        task="reasoning",
        question="Who is the shortest?",
        reference_answer=reference_answer,
        prediction=prediction,
        metadata={
            "category": "comparative_reasoning",
            "difficulty": "easy",
            "is_answerable": True,
        },
    )


def test_evaluate_short_answer_prediction_correct() -> None:
    record = make_record(
        prediction="Nigar is the shortest.",
        reference_answer="Nigar",
    )

    result = evaluate_short_answer_prediction(record)

    assert result.item_id == "reasoning_001_en"
    assert result.short_answer_match == 1
    assert result.normalized_reference == "nigar"


def test_evaluate_short_answer_prediction_incorrect() -> None:
    record = make_record(
        prediction="Aysel is the shortest.",
        reference_answer="Nigar",
    )

    result = evaluate_short_answer_prediction(record)

    assert result.short_answer_match == 0


from src.evaluation.short_answer_match import (
    calculate_short_answer_accuracy,
    evaluate_short_answer_matches,
    summarize_short_answer_matches,
)


def test_calculate_short_answer_accuracy() -> None:
    records = [
        make_record(
            prediction="Nigar is the shortest.",
            reference_answer="Nigar",
        ),
        make_record(
            prediction="Aysel is the shortest.",
            reference_answer="Nigar",
        ),
    ]

    results = evaluate_short_answer_matches(records)

    assert calculate_short_answer_accuracy(results) == 0.5


def test_summarize_short_answer_matches() -> None:
    records = [
        make_record(
            prediction="Nigar is the shortest.",
            reference_answer="Nigar",
        ),
        make_record(
            prediction="Aysel is the shortest.",
            reference_answer="Nigar",
        ),
    ]

    results = evaluate_short_answer_matches(records)
    summary = summarize_short_answer_matches(results)

    assert summary == {
        "total": 2,
        "correct": 1,
        "incorrect": 1,
        "accuracy": 0.5,
    }


from src.evaluation.short_answer_match import (
    filter_short_answer_records,
    is_short_answer_task,
)


def test_is_short_answer_task() -> None:
    assert is_short_answer_task("reasoning") is True
    assert is_short_answer_task("factual_knowledge") is True
    assert is_short_answer_task("linguistic_understanding") is False

    assert is_short_answer_task("instruction_following") is False
    assert is_short_answer_task("unanswerable") is False


def test_filter_short_answer_records() -> None:
    reasoning_record = make_record(
        prediction="Nigar",
        reference_answer="Nigar",
    )

    instruction_record = PredictionRecord(
        item_id="instruction_001_en",
        pair_id="instruction_001",
        language="en",
        task="instruction_following",
        question="Output RESULT: red",
        reference_answer="RESULT: red",
        prediction="RESULT: red",
        metadata={
            "category": "format_following",
            "difficulty": "easy",
            "is_answerable": True,
        },
    )

    filtered = filter_short_answer_records(
        [
            reasoning_record,
            instruction_record,
        ]
    )

    assert len(filtered) == 1
    assert filtered[0].task == "reasoning"
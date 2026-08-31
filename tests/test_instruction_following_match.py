"""Instruction-following evaluator testleri."""

from src.evaluation.instruction_following_match import (
    evaluate_instruction_following,
    evaluate_instruction_following_prediction,
    instruction_following_score,
)
from src.evaluation.run_inference import PredictionRecord
from src.evaluation.instruction_following_match import (
    calculate_instruction_following_accuracy,
    summarize_instruction_following,
)



def make_record(
    prediction: str,
    reference_answer: str,
) -> PredictionRecord:
    return PredictionRecord(
        item_id="instruction_following_001_en",
        pair_id="instruction_following_001",
        language="en",
        task="instruction_following",
        question="Output the answer in the required format.",
        reference_answer=reference_answer,
        prediction=prediction,
        metadata={
            "category": "format_following",
            "difficulty": "easy",
            "is_answerable": True,
        },
    )


def test_exact_instruction_match() -> None:
    assert instruction_following_score(
        "RESULT: red",
        "RESULT: red",
    ) == 1


def test_case_and_punctuation_normalization() -> None:
    assert instruction_following_score(
        "RESULT: RED.",
        "RESULT: red",
    ) == 1


def test_extra_explanation_is_incorrect() -> None:
    assert instruction_following_score(
        "The red car is faster. RESULT: red",
        "RESULT: red",
    ) == 0


def test_wrong_output_is_incorrect() -> None:
    assert instruction_following_score(
        "RESULT: blue",
        "RESULT: red",
    ) == 0


def test_evaluate_instruction_following_prediction() -> None:
    record = make_record(
        prediction="RESULT: red",
        reference_answer="RESULT: red",
    )

    result = evaluate_instruction_following_prediction(
        record
    )

    assert result.item_id == "instruction_following_001_en"
    assert result.instruction_match == 1


def test_evaluate_instruction_following_list() -> None:
    records = [
        make_record(
            prediction="RESULT: red",
            reference_answer="RESULT: red",
        ),
        make_record(
            prediction="RESULT: blue",
            reference_answer="RESULT: red",
        ),
    ]

    results = evaluate_instruction_following(
        records
    )

    assert len(results) == 2
    assert results[0].instruction_match == 1
    assert results[1].instruction_match == 0


def test_calculate_instruction_following_accuracy() -> None:
    records = [
        make_record(
            prediction="RESULT: red",
            reference_answer="RESULT: red",
        ),
        make_record(
            prediction="RESULT: blue",
            reference_answer="RESULT: red",
        ),
    ]

    results = evaluate_instruction_following(
        records
    )

    assert calculate_instruction_following_accuracy(
        results
    ) == 0.5


def test_summarize_instruction_following() -> None:
    records = [
        make_record(
            prediction="RESULT: red",
            reference_answer="RESULT: red",
        ),
        make_record(
            prediction="RESULT: blue",
            reference_answer="RESULT: red",
        ),
    ]

    results = evaluate_instruction_following(
        records
    )

    summary = summarize_instruction_following(
        results
    )

    assert summary == {
        "total": 2,
        "correct": 1,
        "incorrect": 1,
        "accuracy": 0.5,
    }

def test_extraction_allows_reference_inside_prediction() -> None:
    assert instruction_following_score(
        prediction="The extracted values are Leyla and 900 dollars.",
        reference_answer="Leyla and 900 dollars",
        category="extraction",
    ) == 1


def test_extraction_wrong_value_is_incorrect() -> None:
    assert instruction_following_score(
        prediction="The extracted values are Leyla and 800 dollars.",
        reference_answer="Leyla and 900 dollars",
        category="extraction",
    ) == 0


def test_transformation_correct_output() -> None:
    assert instruction_following_score(
        prediction="ARTIFICIAL INTELLIGENCE IS CHANGING TECHNOLOGY",
        reference_answer="ARTIFICIAL INTELLIGENCE IS CHANGING TECHNOLOGY",
        category="transformation",
    ) == 1


def test_transformation_incorrect_output() -> None:
    assert instruction_following_score(
        prediction="Artificial intelligence is changing technology",
        reference_answer="ARTIFICIAL INTELLIGENCE IS CHANGING TECHNOLOGY",
        category="transformation",
    ) == 0


def test_format_following_correct_output() -> None:
    assert instruction_following_score(
        prediction="red\ngreen\nblue",
        reference_answer="red\ngreen\nblue",
        category="format_following",
    ) == 1


def test_format_following_extra_text_is_incorrect() -> None:
    assert instruction_following_score(
        prediction="Colors:\nred\ngreen\nblue",
        reference_answer="red\ngreen\nblue",
        category="format_following",
    ) == 0


def test_constraint_following_correct_output() -> None:
    assert instruction_following_score(
        prediction="AI improves efficiency.",
        reference_answer="AI improves efficiency.",
        category="constraint_following",
    ) == 1


def test_constraint_following_extra_text_is_incorrect() -> None:
    assert instruction_following_score(
        prediction="AI improves efficiency. It also saves time.",
        reference_answer="AI improves efficiency.",
        category="constraint_following",
    ) == 0

def test_multi_step_instruction_correct_output() -> None:
    assert instruction_following_score(
        prediction="RESULT: red",
        reference_answer="RESULT: red",
        category="multi_step_instruction",
    ) == 1


def test_multi_step_instruction_extra_explanation_is_incorrect() -> None:
    assert instruction_following_score(
        prediction="Red is faster.\nRESULT: red",
        reference_answer="RESULT: red",
        category="multi_step_instruction",
    ) == 0
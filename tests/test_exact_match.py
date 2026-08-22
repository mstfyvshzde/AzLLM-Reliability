"""Normalized exact-match capability metriğini test eder."""

import pytest

from src.evaluation.exact_match import (
    ExactMatchResult,
    calculate_accuracy,
    evaluate_exact_match,
    evaluate_prediction,
    exact_match_score,
    summarize_exact_match,
)
from src.evaluation.run_inference import PredictionRecord


def make_prediction_record(
    item_id: str = "reasoning_0001_en",
    pair_id: str = "reasoning_0001",
    language: str = "en",
    prediction: str = "4",
    reference_answer: str = "4",
) -> PredictionRecord:
    """Exact-match testleri için örnek PredictionRecord oluşturur."""

    return PredictionRecord(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task="reasoning",
        question="What is 2 + 2?",
        reference_answer=reference_answer,
        prediction=prediction,
        metadata={
            "category": "arithmetic_reasoning",
            "difficulty": "easy",
            "review_status": "approved",
        },
    )


def test_exact_match_score_for_identical_answers() -> None:
    """Aynı prediction ve reference answer için score=1 döndüğünü test eder."""

    score = exact_match_score(
        prediction="4",
        reference_answer="4",
    )

    assert score == 1


def test_exact_match_score_uses_normalization() -> None:
    """Yüzeysel farkların normalization sonrası eşit kabul edildiğini test eder."""

    score = exact_match_score(
        prediction="  Four. ",
        reference_answer="four",
    )

    assert score == 1


def test_exact_match_score_for_different_answers() -> None:
    """Farklı prediction ve reference answer için score=0 döndüğünü test eder."""

    score = exact_match_score(
        prediction="5",
        reference_answer="4",
    )

    assert score == 0


def test_evaluate_prediction() -> None:
    """Tek PredictionRecord için item-level exact-match sonucu üretildiğini test eder."""

    record = make_prediction_record(
        prediction=" Four. ",
        reference_answer="four",
    )

    result = evaluate_prediction(
        record
    )

    assert isinstance(
        result,
        ExactMatchResult,
    )

    assert result.item_id == record.item_id
    assert result.pair_id == record.pair_id
    assert result.language == record.language
    assert result.task == record.task

    assert result.prediction == " Four. "
    assert result.reference_answer == "four"

    assert result.normalized_prediction == "four"
    assert result.normalized_reference == "four"

    assert result.exact_match == 1
    assert result.metadata == record.metadata


def test_evaluate_prediction_for_incorrect_answer() -> None:
    """Yanlış prediction için exact_match değerinin 0 olduğunu test eder."""

    record = make_prediction_record(
        prediction="5",
        reference_answer="4",
    )

    result = evaluate_prediction(
        record
    )

    assert result.exact_match == 0


def test_exact_match_result_to_dict() -> None:
    """ExactMatchResult nesnesinin sözlüğe doğru dönüştürüldüğünü test eder."""

    result = ExactMatchResult(
        item_id="reasoning_0001_en",
        pair_id="reasoning_0001",
        language="en",
        task="reasoning",
        prediction="Four.",
        reference_answer="four",
        normalized_prediction="four",
        normalized_reference="four",
        exact_match=1,
        metadata={
            "category": "arithmetic_reasoning",
        },
    )

    assert result.to_dict() == {
        "item_id": "reasoning_0001_en",
        "pair_id": "reasoning_0001",
        "language": "en",
        "task": "reasoning",
        "prediction": "Four.",
        "reference_answer": "four",
        "normalized_prediction": "four",
        "normalized_reference": "four",
        "exact_match": 1,
        "metadata": {
            "category": "arithmetic_reasoning",
        },
    }


def test_evaluate_exact_match_preserves_input_order() -> None:
    """Batch evaluation sonucunda prediction sırasının korunduğunu test eder."""

    records = [
        make_prediction_record(
            item_id="reasoning_0002_az",
            pair_id="reasoning_0002",
            language="az",
            prediction="6",
            reference_answer="6",
        ),
        make_prediction_record(
            item_id="reasoning_0001_en",
            pair_id="reasoning_0001",
            language="en",
            prediction="4",
            reference_answer="4",
        ),
    ]

    results = evaluate_exact_match(
        records
    )

    assert [
        result.item_id
        for result in results
    ] == [
        "reasoning_0002_az",
        "reasoning_0001_en",
    ]


def test_evaluate_exact_match_empty_input() -> None:
    """Boş prediction listesinin boş result listesi döndürdüğünü test eder."""

    assert evaluate_exact_match(
        []
    ) == []


def test_calculate_accuracy() -> None:
    """Correct ve incorrect sonuçlardan accuracy hesaplandığını test eder."""

    results = [
        ExactMatchResult(
            item_id="item_1",
            pair_id="pair_1",
            language="en",
            task="reasoning",
            prediction="4",
            reference_answer="4",
            normalized_prediction="4",
            normalized_reference="4",
            exact_match=1,
            metadata={},
        ),
        ExactMatchResult(
            item_id="item_2",
            pair_id="pair_2",
            language="en",
            task="reasoning",
            prediction="5",
            reference_answer="4",
            normalized_prediction="5",
            normalized_reference="4",
            exact_match=0,
            metadata={},
        ),
        ExactMatchResult(
            item_id="item_3",
            pair_id="pair_3",
            language="az",
            task="reasoning",
            prediction="6",
            reference_answer="6",
            normalized_prediction="6",
            normalized_reference="6",
            exact_match=1,
            metadata={},
        ),
    ]

    accuracy = calculate_accuracy(
        results
    )

    assert accuracy == pytest.approx(
        2 / 3
    )


def test_calculate_accuracy_rejects_empty_results() -> None:
    """Boş result listesinde accuracy hesaplanmasının reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Cannot calculate accuracy from empty results",
    ):
        calculate_accuracy(
            []
        )


def test_summarize_exact_match() -> None:
    """Exact-match aggregate özetinin doğru oluşturulduğunu test eder."""

    records = [
        make_prediction_record(
            item_id="reasoning_0001_en",
            pair_id="reasoning_0001",
            language="en",
            prediction="4",
            reference_answer="4",
        ),
        make_prediction_record(
            item_id="reasoning_0001_az",
            pair_id="reasoning_0001",
            language="az",
            prediction="5",
            reference_answer="4",
        ),
        make_prediction_record(
            item_id="reasoning_0002_en",
            pair_id="reasoning_0002",
            language="en",
            prediction="6",
            reference_answer="6",
        ),
    ]

    results = evaluate_exact_match(
        records
    )

    summary = summarize_exact_match(
        results
    )

    assert summary["total"] == 3
    assert summary["correct"] == 2
    assert summary["incorrect"] == 1
    assert summary["accuracy"] == pytest.approx(
        2 / 3
    )


def test_summarize_exact_match_rejects_empty_results() -> None:
    """Boş exact-match sonuçlarının summarize edilmesinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Cannot summarize empty exact-match results",
    ):
        summarize_exact_match(
            []
        )
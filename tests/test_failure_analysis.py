"""Failure-analysis modülünün testleri."""

import pytest

from src.analysis.failure_analysis import (
    CAPABILITY_FAILURE,
    CORRECT,
    FailureAnalysisResult,
    analyze_failure,
    analyze_failures,
    calculate_failure_rate,
    classify_failure,
    summarize_failures,
    summarize_failures_by_language,
    summarize_failures_by_task,
)
from src.evaluation.run_inference import PredictionRecord
from src.reliability.abstention_metrics import (
    EMPTY_RESPONSE,
    OVER_ANSWERING,
    UNDER_ANSWERING,
)


def make_prediction(
    *,
    item_id: str = "reasoning_001_en",
    pair_id: str = "reasoning_001",
    language: str = "en",
    task: str = "reasoning",
    prediction: str = "Nigar",
    reference_answer: str = "Nigar",
) -> PredictionRecord:
    """Testlerde kullanılacak prediction kaydı oluşturur."""

    return PredictionRecord(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        question="Test question",
        prediction=prediction,
        reference_answer=reference_answer,
        metadata={},
    )


def make_failure_result(
    *,
    item_id: str,
    pair_id: str,
    language: str,
    task: str,
    failure_type: str,
) -> FailureAnalysisResult:
    """Testlerde kullanılacak failure sonucu oluşturur."""

    return FailureAnalysisResult(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        failure_type=failure_type,
        metadata={},
    )


def test_classify_over_answering() -> None:
    assert classify_failure(
        capability_correct=None,
        reliability_outcome=OVER_ANSWERING,
    ) == OVER_ANSWERING


def test_classify_under_answering() -> None:
    assert classify_failure(
        capability_correct=0,
        reliability_outcome=UNDER_ANSWERING,
    ) == UNDER_ANSWERING


def test_classify_empty_response() -> None:
    assert classify_failure(
        capability_correct=0,
        reliability_outcome=EMPTY_RESPONSE,
    ) == EMPTY_RESPONSE


def test_classify_capability_failure() -> None:
    assert classify_failure(
        capability_correct=0,
        reliability_outcome="correct_answer",
    ) == CAPABILITY_FAILURE


def test_classify_correct() -> None:
    assert classify_failure(
        capability_correct=1,
        reliability_outcome="correct_answer",
    ) == CORRECT


def test_analyze_failure() -> None:
    record = make_prediction(
        prediction="Leyla",
        reference_answer="Nigar",
    )

    result = analyze_failure(
        record,
        capability_correct=0,
        reliability_outcome="correct_answer",
    )

    assert result.item_id == "reasoning_001_en"
    assert result.pair_id == "reasoning_001"
    assert result.language == "en"
    assert result.task == "reasoning"
    assert result.failure_type == CAPABILITY_FAILURE


def test_analyze_failures() -> None:
    records = [
        make_prediction(
            item_id="reasoning_001_en",
            pair_id="reasoning_001",
            language="en",
            prediction="Leyla",
            reference_answer="Nigar",
        ),
        make_prediction(
            item_id="reasoning_001_az",
            pair_id="reasoning_001",
            language="az",
            prediction="Nigar",
            reference_answer="Nigar",
        ),
    ]

    capability_scores = {
        "reasoning_001_en": 0,
        "reasoning_001_az": 1,
    }

    reliability_outcomes = {
        "reasoning_001_en": "correct_answer",
        "reasoning_001_az": "correct_answer",
    }

    results = analyze_failures(
        records,
        capability_scores=capability_scores,
        reliability_outcomes=reliability_outcomes,
    )

    assert len(results) == 2
    assert results[0].failure_type == CAPABILITY_FAILURE
    assert results[1].failure_type == CORRECT


def test_calculate_failure_rate() -> None:
    results = [
        make_failure_result(
            item_id="item_1",
            pair_id="pair_1",
            language="en",
            task="reasoning",
            failure_type=CORRECT,
        ),
        make_failure_result(
            item_id="item_2",
            pair_id="pair_2",
            language="az",
            task="reasoning",
            failure_type=CAPABILITY_FAILURE,
        ),
        make_failure_result(
            item_id="item_3",
            pair_id="pair_3",
            language="az",
            task="unanswerable",
            failure_type=OVER_ANSWERING,
        ),
        make_failure_result(
            item_id="item_4",
            pair_id="pair_4",
            language="en",
            task="factual_knowledge",
            failure_type=CORRECT,
        ),
    ]

    assert calculate_failure_rate(results) == 0.5


def test_calculate_failure_rate_rejects_empty_results() -> None:
    with pytest.raises(
        ValueError,
        match="Cannot calculate failure rate",
    ):
        calculate_failure_rate([])


def test_summarize_failures_by_language() -> None:
    results = [
        make_failure_result(
            item_id="item_1_en",
            pair_id="pair_1",
            language="en",
            task="reasoning",
            failure_type=CORRECT,
        ),
        make_failure_result(
            item_id="item_2_en",
            pair_id="pair_2",
            language="en",
            task="reasoning",
            failure_type=CAPABILITY_FAILURE,
        ),
        make_failure_result(
            item_id="item_1_az",
            pair_id="pair_1",
            language="az",
            task="reasoning",
            failure_type=CAPABILITY_FAILURE,
        ),
        make_failure_result(
            item_id="item_2_az",
            pair_id="pair_2",
            language="az",
            task="unanswerable",
            failure_type=OVER_ANSWERING,
        ),
    ]

    summary = summarize_failures_by_language(
        results
    )

    assert summary["en"] == {
        "total": 2,
        "correct": 1,
        "capability_failure": 1,
        "over_answering": 0,
        "under_answering": 0,
        "empty_response": 0,
        "failure_rate": 0.5,
    }

    assert summary["az"] == {
        "total": 2,
        "correct": 0,
        "capability_failure": 1,
        "over_answering": 1,
        "under_answering": 0,
        "empty_response": 0,
        "failure_rate": 1.0,
    }


def test_summarize_failures_by_task() -> None:
    results = [
        make_failure_result(
            item_id="item_1",
            pair_id="pair_1",
            language="en",
            task="reasoning",
            failure_type=CORRECT,
        ),
        make_failure_result(
            item_id="item_2",
            pair_id="pair_2",
            language="az",
            task="reasoning",
            failure_type=CAPABILITY_FAILURE,
        ),
        make_failure_result(
            item_id="item_3",
            pair_id="pair_3",
            language="az",
            task="unanswerable",
            failure_type=OVER_ANSWERING,
        ),
    ]

    summary = summarize_failures_by_task(
        results
    )

    assert summary["reasoning"] == {
        "total": 2,
        "correct": 1,
        "capability_failure": 1,
        "over_answering": 0,
        "under_answering": 0,
        "empty_response": 0,
        "failure_rate": 0.5,
    }

    assert summary["unanswerable"] == {
        "total": 1,
        "correct": 0,
        "capability_failure": 0,
        "over_answering": 1,
        "under_answering": 0,
        "empty_response": 0,
        "failure_rate": 1.0,
    }


def test_summarize_failures() -> None:
    results = [
        make_failure_result(
            item_id="item_1_en",
            pair_id="pair_1",
            language="en",
            task="reasoning",
            failure_type=CORRECT,
        ),
        make_failure_result(
            item_id="item_1_az",
            pair_id="pair_1",
            language="az",
            task="reasoning",
            failure_type=CAPABILITY_FAILURE,
        ),
        make_failure_result(
            item_id="item_2_az",
            pair_id="pair_2",
            language="az",
            task="unanswerable",
            failure_type=OVER_ANSWERING,
        ),
        make_failure_result(
            item_id="item_3_en",
            pair_id="pair_3",
            language="en",
            task="factual_knowledge",
            failure_type=UNDER_ANSWERING,
        ),
    ]

    summary = summarize_failures(results)

    assert summary["overall"] == {
        "total": 4,
        "correct": 1,
        "capability_failure": 1,
        "over_answering": 1,
        "under_answering": 1,
        "empty_response": 0,
        "failure_rate": 0.75,
    }

    assert summary["by_language"]["en"]["total"] == 2
    assert summary["by_language"]["az"]["total"] == 2

    assert summary["by_task"]["reasoning"]["total"] == 2
    assert summary["by_task"]["unanswerable"]["total"] == 1
    assert summary["by_task"]["factual_knowledge"]["total"] == 1


def test_summarize_failures_rejects_unknown_failure_type() -> None:
    results = [
        make_failure_result(
            item_id="item_1",
            pair_id="pair_1",
            language="en",
            task="reasoning",
            failure_type="unknown_failure",
        )
    ]

    with pytest.raises(
        ValueError,
        match="Unknown failure type",
    ):
        summarize_failures(results)


def test_failure_analysis_result_to_dict() -> None:
    result = make_failure_result(
        item_id="reasoning_001_az",
        pair_id="reasoning_001",
        language="az",
        task="reasoning",
        failure_type=CAPABILITY_FAILURE,
    )

    assert result.to_dict() == {
        "item_id": "reasoning_001_az",
        "pair_id": "reasoning_001",
        "language": "az",
        "task": "reasoning",
        "failure_type": "capability_failure",
        "metadata": {},
    }
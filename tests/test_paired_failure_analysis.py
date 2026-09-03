"""Paired failure-analysis modülünün testleri."""

import pytest

from src.analysis.failure_analysis import (
    CAPABILITY_FAILURE,
    CORRECT,
    OVER_ANSWERING,
    FailureAnalysisResult,
)
from src.analysis.paired_failure_analysis import (
    LANGUAGE_DEGRADATION,
    LANGUAGE_RECOVERY,
    PRESERVED_BEHAVIOR,
    SHARED_FAILURE,
    PairedFailureAnalysisResult,
    classify_paired_failure_transition,
    evaluate_paired_failure,
    evaluate_paired_failures,
    group_failure_results_by_pair,
    is_correct_failure_result,
    summarize_paired_failures,
    summarize_paired_failures_by_task,
)


def make_failure_result(
    *,
    item_id: str,
    pair_id: str,
    language: str,
    task: str = "reasoning",
    failure_type: str = CORRECT,
) -> FailureAnalysisResult:
    """Testlerde kullanılacak failure-analysis sonucu oluşturur."""

    return FailureAnalysisResult(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        failure_type=failure_type,
        metadata={},
    )


def make_paired_result(
    *,
    pair_id: str,
    task: str = "reasoning",
    transition: str,
    source_correct: int,
    target_correct: int,
) -> PairedFailureAnalysisResult:
    """Testlerde kullanılacak paired failure sonucu oluşturur."""

    return PairedFailureAnalysisResult(
        pair_id=pair_id,
        task=task,
        source_language="en",
        target_language="az",
        source_failure_type=(
            CORRECT
            if source_correct
            else CAPABILITY_FAILURE
        ),
        target_failure_type=(
            CORRECT
            if target_correct
            else CAPABILITY_FAILURE
        ),
        source_correct=source_correct,
        target_correct=target_correct,
        transition=transition,
        metadata={},
    )


def test_is_correct_failure_result() -> None:
    correct_result = make_failure_result(
        item_id="item_1",
        pair_id="pair_1",
        language="en",
        failure_type=CORRECT,
    )

    incorrect_result = make_failure_result(
        item_id="item_2",
        pair_id="pair_2",
        language="en",
        failure_type=CAPABILITY_FAILURE,
    )

    assert is_correct_failure_result(
        correct_result
    ) == 1

    assert is_correct_failure_result(
        incorrect_result
    ) == 0


def test_classify_preserved_behavior() -> None:
    assert classify_paired_failure_transition(
        source_correct=1,
        target_correct=1,
    ) == PRESERVED_BEHAVIOR


def test_classify_language_degradation() -> None:
    assert classify_paired_failure_transition(
        source_correct=1,
        target_correct=0,
    ) == LANGUAGE_DEGRADATION


def test_classify_language_recovery() -> None:
    assert classify_paired_failure_transition(
        source_correct=0,
        target_correct=1,
    ) == LANGUAGE_RECOVERY


def test_classify_shared_failure() -> None:
    assert classify_paired_failure_transition(
        source_correct=0,
        target_correct=0,
    ) == SHARED_FAILURE


def test_classify_rejects_invalid_source_correct() -> None:
    with pytest.raises(
        ValueError,
        match="source_correct must be 0 or 1",
    ):
        classify_paired_failure_transition(
            source_correct=2,
            target_correct=1,
        )


def test_classify_rejects_invalid_target_correct() -> None:
    with pytest.raises(
        ValueError,
        match="target_correct must be 0 or 1",
    ):
        classify_paired_failure_transition(
            source_correct=1,
            target_correct=-1,
        )


def test_group_failure_results_by_pair() -> None:
    results = [
        make_failure_result(
            item_id="pair_1_en",
            pair_id="pair_1",
            language="en",
        ),
        make_failure_result(
            item_id="pair_1_az",
            pair_id="pair_1",
            language="az",
        ),
        make_failure_result(
            item_id="pair_2_en",
            pair_id="pair_2",
            language="en",
        ),
    ]

    grouped = group_failure_results_by_pair(
        results
    )

    assert set(grouped) == {
        "pair_1",
        "pair_2",
    }

    assert len(grouped["pair_1"]) == 2
    assert len(grouped["pair_2"]) == 1


def test_evaluate_paired_failure_preserved() -> None:
    pair_results = [
        make_failure_result(
            item_id="reasoning_001_en",
            pair_id="reasoning_001",
            language="en",
            failure_type=CORRECT,
        ),
        make_failure_result(
            item_id="reasoning_001_az",
            pair_id="reasoning_001",
            language="az",
            failure_type=CORRECT,
        ),
    ]

    result = evaluate_paired_failure(
        pair_results,
        source_language="en",
        target_language="az",
    )

    assert result.pair_id == "reasoning_001"
    assert result.task == "reasoning"
    assert result.source_correct == 1
    assert result.target_correct == 1
    assert result.transition == PRESERVED_BEHAVIOR


def test_evaluate_paired_failure_degradation() -> None:
    pair_results = [
        make_failure_result(
            item_id="reasoning_001_en",
            pair_id="reasoning_001",
            language="en",
            failure_type=CORRECT,
        ),
        make_failure_result(
            item_id="reasoning_001_az",
            pair_id="reasoning_001",
            language="az",
            failure_type=CAPABILITY_FAILURE,
        ),
    ]

    result = evaluate_paired_failure(
        pair_results,
        source_language="en",
        target_language="az",
    )

    assert result.source_correct == 1
    assert result.target_correct == 0
    assert result.transition == LANGUAGE_DEGRADATION


def test_evaluate_paired_failure_keeps_failure_types() -> None:
    pair_results = [
        make_failure_result(
            item_id="unanswerable_001_en",
            pair_id="unanswerable_001",
            language="en",
            task="unanswerable",
            failure_type=CORRECT,
        ),
        make_failure_result(
            item_id="unanswerable_001_az",
            pair_id="unanswerable_001",
            language="az",
            task="unanswerable",
            failure_type=OVER_ANSWERING,
        ),
    ]

    result = evaluate_paired_failure(
        pair_results,
        source_language="en",
        target_language="az",
    )

    assert result.source_failure_type == CORRECT
    assert result.target_failure_type == OVER_ANSWERING
    assert result.transition == LANGUAGE_DEGRADATION


def test_evaluate_paired_failure_rejects_missing_source() -> None:
    pair_results = [
        make_failure_result(
            item_id="reasoning_001_az",
            pair_id="reasoning_001",
            language="az",
        )
    ]

    with pytest.raises(
        ValueError,
        match="Missing source-language result",
    ):
        evaluate_paired_failure(
            pair_results,
            source_language="en",
            target_language="az",
        )


def test_evaluate_paired_failure_rejects_missing_target() -> None:
    pair_results = [
        make_failure_result(
            item_id="reasoning_001_en",
            pair_id="reasoning_001",
            language="en",
        )
    ]

    with pytest.raises(
        ValueError,
        match="Missing target-language result",
    ):
        evaluate_paired_failure(
            pair_results,
            source_language="en",
            target_language="az",
        )


def test_evaluate_paired_failure_rejects_inconsistent_tasks() -> None:
    pair_results = [
        make_failure_result(
            item_id="item_en",
            pair_id="pair_1",
            language="en",
            task="reasoning",
        ),
        make_failure_result(
            item_id="item_az",
            pair_id="pair_1",
            language="az",
            task="factual_knowledge",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="inconsistent tasks",
    ):
        evaluate_paired_failure(
            pair_results,
            source_language="en",
            target_language="az",
        )


def test_evaluate_paired_failures() -> None:
    results = [
        make_failure_result(
            item_id="pair_1_en",
            pair_id="pair_1",
            language="en",
            failure_type=CORRECT,
        ),
        make_failure_result(
            item_id="pair_1_az",
            pair_id="pair_1",
            language="az",
            failure_type=CAPABILITY_FAILURE,
        ),
        make_failure_result(
            item_id="pair_2_en",
            pair_id="pair_2",
            language="en",
            failure_type=CAPABILITY_FAILURE,
        ),
        make_failure_result(
            item_id="pair_2_az",
            pair_id="pair_2",
            language="az",
            failure_type=CORRECT,
        ),
    ]

    paired = evaluate_paired_failures(
        results,
        source_language="en",
        target_language="az",
    )

    assert len(paired) == 2

    assert paired[0].transition == LANGUAGE_DEGRADATION
    assert paired[1].transition == LANGUAGE_RECOVERY


def test_summarize_paired_failures_by_task() -> None:
    results = [
        make_paired_result(
            pair_id="pair_1",
            task="reasoning",
            transition=PRESERVED_BEHAVIOR,
            source_correct=1,
            target_correct=1,
        ),
        make_paired_result(
            pair_id="pair_2",
            task="reasoning",
            transition=LANGUAGE_DEGRADATION,
            source_correct=1,
            target_correct=0,
        ),
        make_paired_result(
            pair_id="pair_3",
            task="linguistic_understanding",
            transition=SHARED_FAILURE,
            source_correct=0,
            target_correct=0,
        ),
    ]

    summary = summarize_paired_failures_by_task(
        results
    )

    assert summary["reasoning"] == {
        "total_pairs": 2,
        "preserved_behavior": 1,
        "language_degradation": 1,
        "language_recovery": 0,
        "shared_failure": 0,
        "behavior_preservation_rate": 0.5,
        "degradation_rate": 0.5,
        "recovery_rate": 0.0,
        "shared_failure_rate": 0.0,
    }

    assert summary[
        "linguistic_understanding"
    ]["shared_failure"] == 1


def test_summarize_paired_failures() -> None:
    results = [
        make_paired_result(
            pair_id="pair_1",
            transition=PRESERVED_BEHAVIOR,
            source_correct=1,
            target_correct=1,
        ),
        make_paired_result(
            pair_id="pair_2",
            transition=LANGUAGE_DEGRADATION,
            source_correct=1,
            target_correct=0,
        ),
        make_paired_result(
            pair_id="pair_3",
            transition=LANGUAGE_RECOVERY,
            source_correct=0,
            target_correct=1,
        ),
        make_paired_result(
            pair_id="pair_4",
            transition=SHARED_FAILURE,
            source_correct=0,
            target_correct=0,
        ),
    ]

    summary = summarize_paired_failures(
        results
    )

    assert summary["overall"] == {
        "total_pairs": 4,
        "preserved_behavior": 1,
        "language_degradation": 1,
        "language_recovery": 1,
        "shared_failure": 1,
        "behavior_preservation_rate": 0.25,
        "degradation_rate": 0.25,
        "recovery_rate": 0.25,
        "shared_failure_rate": 0.25,
    }


def test_summarize_empty_paired_failures() -> None:
    summary = summarize_paired_failures([])

    assert summary["overall"] == {
        "total_pairs": 0,
        "preserved_behavior": 0,
        "language_degradation": 0,
        "language_recovery": 0,
        "shared_failure": 0,
        "behavior_preservation_rate": None,
        "degradation_rate": None,
        "recovery_rate": None,
        "shared_failure_rate": None,
    }


def test_paired_failure_result_to_dict() -> None:
    result = make_paired_result(
        pair_id="reasoning_001",
        transition=LANGUAGE_DEGRADATION,
        source_correct=1,
        target_correct=0,
    )

    data = result.to_dict()

    assert data["pair_id"] == "reasoning_001"
    assert data["source_language"] == "en"
    assert data["target_language"] == "az"
    assert data["source_correct"] == 1
    assert data["target_correct"] == 0
    assert data["transition"] == LANGUAGE_DEGRADATION
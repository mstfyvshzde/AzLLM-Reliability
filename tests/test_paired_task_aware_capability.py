"""Paired task-aware capability evaluator testleri."""

from src.evaluation.paired_task_aware_capability import (
    BOTH_CORRECT,
    BOTH_INCORRECT,
    SOURCE_ONLY_CORRECT,
    TARGET_ONLY_CORRECT,
    classify_task_aware_transition,
    evaluate_paired_task_aware_capability,
)
from src.evaluation.task_aware_capability import (
    TaskAwareCapabilityResult,
)


def make_result(
    language: str,
    correct: int,
) -> TaskAwareCapabilityResult:
    return TaskAwareCapabilityResult(
        item_id=f"reasoning_001_{language}",
        pair_id="reasoning_001",
        language=language,
        task="reasoning",
        prediction="Nigar",
        reference_answer="Nigar",
        evaluator="short_answer",
        correct=correct,
        metadata={
            "category": "comparative_reasoning",
            "difficulty": "easy",
            "is_answerable": True,
        },
    )


def test_classify_both_correct() -> None:
    assert classify_task_aware_transition(
        1,
        1,
    ) == BOTH_CORRECT


def test_classify_source_only_correct() -> None:
    assert classify_task_aware_transition(
        1,
        0,
    ) == SOURCE_ONLY_CORRECT


def test_classify_target_only_correct() -> None:
    assert classify_task_aware_transition(
        0,
        1,
    ) == TARGET_ONLY_CORRECT


def test_classify_both_incorrect() -> None:
    assert classify_task_aware_transition(
        0,
        0,
    ) == BOTH_INCORRECT


def test_evaluate_paired_task_aware_capability() -> None:
    results = [
        make_result(
            language="en",
            correct=1,
        ),
        make_result(
            language="az",
            correct=0,
        ),
    ]

    paired = evaluate_paired_task_aware_capability(
        results
    )

    assert len(paired) == 1

    result = paired[0]

    assert result.pair_id == "reasoning_001"
    assert result.source_correct == 1
    assert result.target_correct == 0
    assert result.transition == SOURCE_ONLY_CORRECT


from src.evaluation.paired_task_aware_capability import (
    summarize_paired_task_aware_capability,
)


def test_summarize_paired_task_aware_capability() -> None:
    results = [
        make_result(
            language="en",
            correct=1,
        ),
        make_result(
            language="az",
            correct=0,
        ),
    ]

    paired = evaluate_paired_task_aware_capability(
        results
    )

    summary = summarize_paired_task_aware_capability(
        paired
    )

    assert summary == {
        "total_pairs": 1,
        "both_correct": 0,
        "source_only_correct": 1,
        "target_only_correct": 0,
        "both_incorrect": 0,
        "degradation_rate": 1.0,
        "recovery_rate": 0.0,
        "consistency_rate": 0.0,
    }
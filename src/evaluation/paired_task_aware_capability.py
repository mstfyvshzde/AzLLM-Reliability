"""Task-aware capability sonuçlarını EN-AZ pair seviyesinde karşılaştırır.

Her semantic pair için source ve target dil sonuçları birlikte değerlendirilir.

Transition sınıfları:

    both_correct
        Her iki dilde de doğru.

    source_only_correct
        Source doğru, target yanlış.

    target_only_correct
        Source yanlış, target doğru.

    both_incorrect
        Her iki dilde de yanlış.
"""


from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.task_aware_capability import (
    TaskAwareCapabilityResult
)


BOTH_CORRECT = "both_correct"
SOURCE_ONLY_CORRECT = "source_only_correct"
TARGET_ONLY_CORRECT = "target_only_correct"
BOTH_INCORRECT = "both_incorrect"


VALID_TRANSITIONS = {
    BOTH_CORRECT,
    SOURCE_ONLY_CORRECT,
    TARGET_ONLY_CORRECT,
    BOTH_INCORRECT
}


@dataclass(frozen=True)
class PairedTaskAwareCapabilityResult:
    """Tek bir EN-AZ pair için task-aware capability sonucunu temsil eder."""

    pair_id: str
    task: str
    source_language: str
    target_language: str
    source_correct: int
    target_correct: int
    transition: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Result nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)



def classify_task_aware_transition(
    source_correct: int,
    target_correct: int
) -> str:
    """Source-target correctness kombinasyonunu transition sınıfına dönüştürür."""

    if source_correct == 1 and target_correct == 1:
        return BOTH_CORRECT

    if source_correct == 1 and target_correct == 0:
        return SOURCE_ONLY_CORRECT

    if source_correct == 0 and target_correct == 1:
        return TARGET_ONLY_CORRECT

    return BOTH_INCORRECT


def evaluate_paired_task_aware_capability(
    results: list[TaskAwareCapabilityResult],
    source_language: str = 'en',
    target_language: str = 'az'
) -> list[PairedTaskAwareCapabilityResult]:
    """Task-aware capability sonuçlarını pair seviyesinde karşılaştırır."""

    grouped: dict[
        str, 
        dict[str, TaskAwareCapabilityResult]
    ] = {}

    for result in results:
        grouped.setdefault(
            result.pair_id,
            {}
        )[result.language] = result

    paired_results: list[
        PairedTaskAwareCapabilityResult
    ] = []

    for pair_id, language_results in grouped.items():
        if source_language not in language_results:
            raise ValueError(
                f"Missing source language '{source_language}' "
                f"for pair '{pair_id}'."
            )

        if target_language not in language_results:
            raise ValueError(
                f"Missing target language '{target_language}' "
                f"for pair '{pair_id}'."
            )

        source_result = language_results[source_language]
        target_result = language_results[target_language]

        if source_result.task != target_result.task:
            raise ValueError(
                f"Task mismatch for pair '{pair_id}'."
            )

        transition = classify_task_aware_transition(
            source_result.correct,
            target_result.correct
        )

        paired_results.append(
            PairedTaskAwareCapabilityResult(
                pair_id=pair_id,
                task=source_result.task,
                source_language=source_language,
                target_language=target_language,
                source_correct=source_result.correct,
                target_correct=target_result.correct,
                transition=transition,
                metadata=dict(source_result.metadata)
            )
        )

    return paired_results



def summarize_paired_task_aware_capability(
    results: list[PairedTaskAwareCapabilityResult]
) -> dict[str, Any]:
    """Paired task-aware capability sonuçlarını özetler."""

    if not results:
        raise ValueError(
            "Cannot summarize empty paired task-aware results."
        )

    counts = {
        BOTH_CORRECT: 0,
        SOURCE_ONLY_CORRECT: 0,
        TARGET_ONLY_CORRECT: 0,
        BOTH_INCORRECT: 0
    }

    for result in results:
        if result.transition not in VALID_TRANSITIONS:
            raise ValueError(
                f"Unknown transition: '{result.transition}'"
            )

        counts[result.transition] += 1

    total = len(results)


    return {
        "total_pairs": total,
        "both_correct": counts[BOTH_CORRECT],
        "source_only_correct": counts[SOURCE_ONLY_CORRECT],
        "target_only_correct": counts[TARGET_ONLY_CORRECT],
        "both_incorrect": counts[BOTH_INCORRECT],
        "degradation_rate": (
            counts[SOURCE_ONLY_CORRECT]
            / total
        ),
        "recovery_rate": (
            counts[TARGET_ONLY_CORRECT]
            / total
        ),
        "consistency_rate": (
            counts[BOTH_CORRECT]
            + counts[BOTH_INCORRECT]
        )
        / total
    }
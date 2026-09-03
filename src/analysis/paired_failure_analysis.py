"""EN -> AZ pair-level failure transition'larını analiz eder.

Bu modül aynı pair_id'ye ait English ve Azerbaijani prediction
sonuçlarını karşılaştırarak capability'nin dil değişiminde nasıl
davrandığını sınıflandırır.

Transition türleri:

    preserved_behavior
        -> EN doğru, AZ doğru.

    language_degradation
        -> EN doğru, AZ yanlış.

    language_recovery
        -> EN yanlış, AZ doğru.

    shared_failure
        -> EN yanlış, AZ yanlış.

Bu analiz prediction-level failure analysis'ten farklıdır.
Burada temel araştırma sorusu şudur:

    Aynı semantik task English'ten Azerbaijani'ye geçtiğinde
    capability korunuyor mu, düşüyor mu, yoksa iyileşiyor mu?
"""


from __future__ import annotations 

from dataclasses import asdict, dataclass
from typing import Any

from src.analysis.failure_analysis import FailureAnalysisResult



PRESERVED_BEHAVIOR = "preserved_behavior"
LANGUAGE_DEGRADATION = "language_degradation"
LANGUAGE_RECOVERY = "language_recovery"
SHARED_FAILURE = "shared_failure"

VALID_PAIRED_FAILURE_TRANSITIONS = {
    PRESERVED_BEHAVIOR,
    LANGUAGE_DEGRADATION,
    LANGUAGE_RECOVERY,
    SHARED_FAILURE
}


@dataclass(frozen=True)
class PairedFailureAnalysisResult:
    """Tek bir EN/AZ pair için failure transition sonucunu temsil eder."""

    pair_id: str
    task: str
    source_language: str
    target_language: str
    source_failure_type: str
    target_failure_type: str
    source_correct: int
    target_correct: int
    transition: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Sonucu JSON-serializable dictionary biçimine dönüştürür."""

        return asdict(self)



def is_correct_failure_result(
    result: FailureAnalysisResult
) -> int:
    """Failure-analysis sonucunun correct olup olmadığını döndürür."""

    return int(
        result.failure_type == 'correct'
    )



def classify_paired_failure_transition(
    *,
    source_correct: int,
    target_correct: int
) -> str:
    """EN ve AZ correctness durumundan pair transition türünü belirler."""

    if source_correct not in {0, 1}:
        raise ValueError(
            "source_correct must be 0 or 1."
        )

    if target_correct not in {0, 1}:
        raise ValueError(
            "target_correct must be 0 or 1."
        )

    if source_correct == 1 and target_correct == 1:
        return PRESERVED_BEHAVIOR

    if source_correct == 1 and target_correct == 0:
        return LANGUAGE_DEGRADATION

    if source_correct == 0 and target_correct == 1:
        return LANGUAGE_RECOVERY

    return SHARED_FAILURE


def group_failure_results_by_pair(
    results: list[FailureAnalysisResult]
) -> dict[str, list[FailureAnalysisResult]]:
    """Failure-analysis sonuçlarını pair_id bazında gruplar."""

    grouped: dict[
        str,
        list[FailureAnalysisResult]
    ] = {}

    for result in results:
        grouped.setdefault(
            result.pair_id,
            []
        ).append(result)

    return grouped


def evaluate_paired_failure(
    pair_results: list[FailureAnalysisResult],
    *,
    source_language: str,
    target_language: str
) -> PairedFailureAnalysisResult:
    """Tek bir pair için language-transition failure analizi yapar."""

    source_result = next(
        (
            result
            for result in pair_results
            if result.language == source_language
        ),
        None
    )

    target_result = next(
        (
            result 
            for result in pair_results
            if result.language == target_language
        ),
        None
    )

    if source_result is None:
        raise ValueError(
            "Missing source-language result for pair."
        )

    if target_result is None:
        raise ValueError(
            "Missing target-language result for pair."
        )

    if source_result.task != target_result.task:
        raise ValueError(
            "Pair contains inconsistent tasks: "
            f"{source_result.task!r} vs "
            f"{target_result.task!r}"
        )

    source_correct = is_correct_failure_result(source_result)
    target_correct = is_correct_failure_result(target_result)

    transition = classify_paired_failure_transition(
        source_correct=source_correct,
        target_correct=target_correct
    )
    

    metadata = {
        "source_metadata": dict(
            source_result.metadata
        ),
        "target_metadata": dict(
            target_result.metadata
        )
    }

    return PairedFailureAnalysisResult(
        pair_id=source_result.pair_id,
        task=source_result.task,
        source_language=source_language,
        target_language=target_language,
        source_failure_type=source_result.failure_type,
        target_failure_type=target_result.failure_type,
        source_correct=source_correct,
        target_correct=target_correct,
        transition=transition,
        metadata=metadata
    )



def evaluate_paired_failures(
    results: list[FailureAnalysisResult],
    *,
    source_language: str,
    target_language: str
) -> list[PairedFailureAnalysisResult]:
    """Tüm failure-analysis sonuçlarından pair-level sonuçlar üretir."""

    grouped = group_failure_results_by_pair(results)

    paired_results: list[PairedFailureAnalysisResult] = []

    for pair_id in sorted(grouped):
        pair_result = evaluate_paired_failure(
            grouped[pair_id],
            source_language=source_language,
            target_language=target_language
        )

        paired_results.append(pair_result)

    return paired_results


def _count_transitions(
    results: list[PairedFailureAnalysisResult]
) -> dict[str, int]:
    """Pair transition türlerinin sayısını hesaplar."""

    counts = {
        PRESERVED_BEHAVIOR: 0,
        LANGUAGE_DEGRADATION: 0,
        LANGUAGE_RECOVERY: 0,
        SHARED_FAILURE: 0
    }

    for result in results:
        if (
            result.transition
            not in VALID_PAIRED_FAILURE_TRANSITIONS
        ):
            raise ValueError(
                "Unknown paired failure transition: "
                f"{result.transition}"
            )

        counts[result.transition] += 1

    return counts



def _summarize_paired_group(
    results: list[PairedFailureAnalysisResult]
) -> dict[str, int | float | None]:
    """Tek bir paired-result grubunun özetini oluşturur."""

    counts = _count_transitions(
        results
    )

    total_pairs = len(results)

    if total_pairs == 0:
        degradation_rate = None
        recovery_rate = None
        behavior_preservation_rate = None
        shared_failure_rate = None

    else:
        degradation_rate = (
            counts[LANGUAGE_DEGRADATION]
            / total_pairs
        )

        recovery_rate = (
            counts[LANGUAGE_RECOVERY]
            / total_pairs
        )

        behavior_preservation_rate = (
            counts[PRESERVED_BEHAVIOR]
            / total_pairs
        )

        shared_failure_rate = (
            counts[SHARED_FAILURE]
            / total_pairs
        )

    return {
        "total_pairs": total_pairs,
        "preserved_behavior": counts[
            PRESERVED_BEHAVIOR
        ],
        "language_degradation": counts[
            LANGUAGE_DEGRADATION
        ],
        "language_recovery": counts[
            LANGUAGE_RECOVERY
        ],
        "shared_failure": counts[
            SHARED_FAILURE
        ],
        "behavior_preservation_rate": behavior_preservation_rate,
        "degradation_rate": degradation_rate,
        "recovery_rate": recovery_rate,
        "shared_failure_rate": shared_failure_rate
    }


def summarize_paired_failures_by_task(
    results: list[PairedFailureAnalysisResult]
) -> dict[str, dict[str, int | float | None]]:
    """Pair-level failure sonuçlarını task bazında özetler."""

    grouped: dict[
        str,
        list[PairedFailureAnalysisResult],
    ] = {}

    for result in results:
        grouped.setdefault(
            result.task,
            []
        ).append(result)

    return {
        task: _summarize_paired_group(
            task_results
        )
        for task, task_results
        in sorted(grouped.items())
    }


def summarize_paired_failures(
    results: list[PairedFailureAnalysisResult]
) -> dict[str, Any]:
    """Pair-level failure sonuçlarının kapsamlı özetini oluşturur."""

    return {
        "overall": _summarize_paired_group(
            results
        ),
        "by_task": summarize_paired_failures_by_task(
            results
        )
    }
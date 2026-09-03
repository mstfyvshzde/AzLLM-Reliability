"""Model prediction'larındaki temel failure türlerini analiz eder.

Bu modül capability ve reliability sonuçlarını birlikte kullanarak
her prediction için temel failure kategorisini belirler.

Failure türleri:

    correct
        -> Prediction capability ve reliability açısından doğru.

    capability_failure
        -> Model answerable bir soruya cevap verdi ancak içerik yanlış.

    over_answering
        -> Model cevaplanamayan bir soruya cevap vermeye çalıştı.

    incorrect_abstention
        -> Model cevaplanabilir bir sorudan gereksiz şekilde kaçındı.

Bu modül ayrıca failure sonuçlarını overall, language ve task bazında
özetlemek için yardımcı fonksiyonlar sağlar.

Not:
    EN -> AZ language degradation pair-level bir olaydır. Bu nedenle
    burada tek prediction failure türü olarak sınıflandırılmaz.
    Pair-level degradation ayrı paired failure analysis katmanında
    değerlendirilecektir.
"""


from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.run_inference import PredictionRecord
from src.reliability.abstention_metrics import (
    EMPTY_RESPONSE,
    OVER_ANSWERING,
    UNDER_ANSWERING
)


CORRECT = "correct"
CAPABILITY_FAILURE = "capability_failure"

VALID_FAILURE_TYPES = {
    CORRECT,
    CAPABILITY_FAILURE,
    OVER_ANSWERING,
    UNDER_ANSWERING,
    EMPTY_RESPONSE
}



@dataclass(frozen=True)
class FailureAnalysisResult:
    """Tek bir prediction için failure-analysis sonucunu temsil eder."""

    item_id: str
    pair_id: str
    language: str
    task: str
    failure_type: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Sonucu JSON-serializable dictionary biçimine dönüştürür."""

        return asdict(self)


def classify_failure(
    *,
    capability_correct: int | None,
    reliability_outcome: str | None,
) -> str:
    """Capability ve reliability sinyallerinden failure türünü belirler.

    Reliability failure'ları capability failure'larından önce ele alınır.

    Args:
        capability_correct:
            Capability evaluator sonucu.
            1 doğru, 0 yanlış, None capability dışı item anlamına gelir.

        reliability_outcome:
            Abstention evaluator tarafından üretilen outcome.
    """

    if reliability_outcome == OVER_ANSWERING:
        return OVER_ANSWERING

    if reliability_outcome == UNDER_ANSWERING:
        return UNDER_ANSWERING

    if reliability_outcome == EMPTY_RESPONSE:
        return EMPTY_RESPONSE

    if capability_correct == 0:
        return CAPABILITY_FAILURE

    return CORRECT




def analyze_failure(
    record: PredictionRecord,
    *,
    capability_correct: int | None,
    reliability_outcome: str | None,
) -> FailureAnalysisResult:
    """Tek bir prediction için failure-analysis sonucu üretir."""

    failure_type = classify_failure(
        capability_correct=capability_correct,
        reliability_outcome=reliability_outcome,
    )

    return FailureAnalysisResult(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        failure_type=failure_type,
        metadata=dict(record.metadata),
    )


def analyze_failures(
    records: list[PredictionRecord],
    *,
    capability_scores: dict[str, int | None],
    reliability_outcomes: dict[str, str | None],
) -> list[FailureAnalysisResult]:
    """Prediction listesinin failure-analysis sonuçlarını üretir."""

    results: list[FailureAnalysisResult] = []

    for record in records:
        results.append(
            analyze_failure(
                record,
                capability_correct=capability_scores.get(
                    record.item_id
                ),
                reliability_outcome=reliability_outcomes.get(
                    record.item_id
                ),
            )
        )

    return results


def calculate_failure_rate(
    results: list[FailureAnalysisResult],
) -> float:
    """Toplam failure oranını hesaplar."""

    if not results:
        raise ValueError(
            "Cannot calculate failure rate from empty results."
        )

    failure_count = sum(
        result.failure_type != CORRECT
        for result in results
    )

    return failure_count / len(results)


def _count_failure_types(
    results: list[FailureAnalysisResult],
) -> dict[str, int]:
    """Failure türlerinin sayısını hesaplar."""

    counts = {
        CORRECT: 0,
        CAPABILITY_FAILURE: 0,
        OVER_ANSWERING: 0,
        UNDER_ANSWERING: 0,
        EMPTY_RESPONSE: 0,
    }

    for result in results:
        if result.failure_type not in VALID_FAILURE_TYPES:
            raise ValueError(
                "Unknown failure type: "
                f"{result.failure_type}"
            )

        counts[result.failure_type] += 1

    return counts


def _summarize_group(
    results: list[FailureAnalysisResult],
) -> dict[str, int | float | None]:
    """Tek bir result grubunun failure özetini oluşturur."""

    counts = _count_failure_types(results)
    total = len(results)

    failures = (
        counts[CAPABILITY_FAILURE]
        + counts[OVER_ANSWERING]
        + counts[UNDER_ANSWERING]
        + counts[EMPTY_RESPONSE]
    )

    failure_rate = (
        failures / total
        if total
        else None
    )

    return {
        "total": total,
        "correct": counts[CORRECT],
        "capability_failure": counts[
            CAPABILITY_FAILURE
        ],
        "over_answering": counts[
            OVER_ANSWERING
        ],
        "under_answering": counts[
            UNDER_ANSWERING
        ],
        "empty_response": counts[
            EMPTY_RESPONSE
        ],
        "failure_rate": failure_rate,
    }


def summarize_failures_by_language(
    results: list[FailureAnalysisResult],
) -> dict[str, dict[str, int | float | None]]:
    """Failure sonuçlarını language bazında özetler."""

    grouped: dict[
        str,
        list[FailureAnalysisResult],
    ] = {}

    for result in results:
        grouped.setdefault(
            result.language,
            [],
        ).append(result)

    return {
        language: _summarize_group(
            language_results
        )
        for language, language_results
        in sorted(grouped.items())
    }


def summarize_failures_by_task(
    results: list[FailureAnalysisResult],
) -> dict[str, dict[str, int | float | None]]:
    """Failure sonuçlarını task bazında özetler."""

    grouped: dict[
        str,
        list[FailureAnalysisResult],
    ] = {}

    for result in results:
        grouped.setdefault(
            result.task,
            [],
        ).append(result)

    return {
        task: _summarize_group(
            task_results
        )
        for task, task_results
        in sorted(grouped.items())
    }


def summarize_failures(
    results: list[FailureAnalysisResult],
) -> dict[str, Any]:
    """Failure-analysis sonuçlarının kapsamlı özetini oluşturur."""

    return {
        "overall": _summarize_group(
            results
        ),
        "by_language": summarize_failures_by_language(
            results
        ),
        "by_task": summarize_failures_by_task(
            results
        ),
    }
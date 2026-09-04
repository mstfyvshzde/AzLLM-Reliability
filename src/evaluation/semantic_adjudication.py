"""Açık semantik cevaplar için adjudication altyapısı.

Bu modül embedding cosine similarity veya test-specific alias kuralları
kullanmaz.

Amaç:
    contextual_meaning
    lexical_disambiguation
    discourse_understanding

kategorilerindeki açık uçlu cevapların önceden tanımlanmış bir rubric ile
değerlendirilmesini desteklemektir.

Adjudication rubric benchmark TEST sonuçları değerlendirilmeden önce
DEV split üzerinde sabitlenmelidir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.normalize_answer import normalize_answer
from src.evaluation.run_inference import PredictionRecord


SEMANTIC_ADJUDICATION_CATEGORIES = {
    "contextual_meaning",
    "lexical_disambiguation",
    "discourse_understanding",
}


@dataclass(frozen=True)
class SemanticAdjudicationResult:
    """Tek bir açık-semantic prediction için adjudication sonucudur."""

    item_id: str
    pair_id: str
    language: str
    task: str
    category: str
    question: str
    prediction: str
    reference_answer: str
    decision: int
    reason: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Sonucu serializable sözlüğe dönüştürür."""
        return asdict(self)


def is_semantic_adjudication_record(
    record: PredictionRecord,
) -> bool:
    """Record açık semantic adjudication gerektiriyor mu kontrol eder."""

    return (
        record.task == "linguistic_understanding"
        and record.metadata.get("category")
        in SEMANTIC_ADJUDICATION_CATEGORIES
    )


def exact_semantic_match(
    prediction: str,
    reference_answer: str,
) -> bool:
    """Normalize edilmiş tam eşleşmeyi güvenli otomatik kabul eder."""

    return (
        normalize_answer(prediction)
        == normalize_answer(reference_answer)
    )


def validate_adjudication_decision(
    decision: int,
) -> None:
    """Adjudication decision yalnızca binary olabilir."""

    if decision not in {0, 1}:
        raise ValueError(
            "Semantic adjudication decision must be 0 or 1."
        )


def create_semantic_adjudication_result(
    record: PredictionRecord,
    decision: int,
    reason: str,
) -> SemanticAdjudicationResult:
    """Manuel semantic adjudication sonucunu oluşturur."""

    if not is_semantic_adjudication_record(record):
        raise ValueError(
            "Record is not eligible for semantic adjudication."
        )

    validate_adjudication_decision(decision)

    cleaned_reason = reason.strip()

    if not cleaned_reason:
        raise ValueError(
            "Semantic adjudication reason cannot be empty."
        )

    return SemanticAdjudicationResult(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        category=str(
            record.metadata.get("category")
        ),
        question=record.question,
        prediction=record.prediction,
        reference_answer=record.reference_answer,
        decision=decision,
        reason=cleaned_reason,
        metadata=dict(record.metadata),
    )


def load_semantic_adjudication_decisions(
    path: str,
) -> dict[str, int]:
    """JSONL adjudication artifact'ını item_id -> decision mapping olarak yükler."""

    import json
    from pathlib import Path

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Semantic adjudication artifact not found: {input_path}"
        )

    decisions: dict[str, int] = {}

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            row = json.loads(stripped)

            item_id = row.get("item_id")
            decision = row.get("decision")

            if not isinstance(item_id, str) or not item_id:
                raise ValueError(
                    "Semantic adjudication item_id must be "
                    f"a non-empty string at line {line_number}."
                )

            validate_adjudication_decision(decision)

            if item_id in decisions:
                raise ValueError(
                    "Duplicate semantic adjudication item_id: "
                    f"{item_id}"
                )

            decisions[item_id] = decision

    if not decisions:
        raise ValueError(
            "Semantic adjudication artifact contains no decisions."
        )

    return decisions

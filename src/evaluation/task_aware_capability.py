"""Task türüne göre uygun capability evaluator sonucunu birleştirir.

Bu modül her task için aynı metriği zorlamak yerine task-aware scoring kullanır.

Current policy:

    factual_knowledge
        -> short_answer_match

    reasoning
        -> short_answer_match

    linguistic_understanding
        -> category-aware:
           reference_resolution -> short_answer_match
           paraphrase_understanding -> option_match
           contextual_meaning / lexical_disambiguation / discourse_understanding -> semantic_adjudication

    instruction_following
        -> category-aware instruction_following_match

    unanswerable
        -> capability scoring dışında tutulur
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.run_inference import PredictionRecord
from src.evaluation.semantic_adjudication import (
    SEMANTIC_ADJUDICATION_CATEGORIES,
)
from src.evaluation.short_answer_match import (
    short_answer_match_score
)
from src.evaluation.binary_answer_match import (
    binary_answer_match_score,
    is_binary_reference,
)
from src.evaluation.instruction_following_match import (
    instruction_following_score
)
from src.evaluation.option_match import (
    option_match_score
)


@dataclass(frozen=True)
class TaskAwareCapabilityResult:
    """Tek bir prediction için task-aware capability sonucunu temsil eder."""

    item_id: str
    pair_id: str
    language: str
    task: str
    prediction: str
    reference_answer: str
    evaluator: str
    correct: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Result nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)


def evaluate_task_aware_prediction(
    record: PredictionRecord,
    semantic_adjudication_decisions: dict[str, int] | None = None,
    binary_adjudication_decisions: dict[str, int] | None = None,
) -> TaskAwareCapabilityResult | None:
    """Prediction'ı task türüne uygun capability evaluator ile değerlendirir.

    Her task aynı şekilde değerlendirilemez. Bu nedenle record.task değerine
    bakarak uygun evaluator seçilir ve prediction ile reference_answer
    karşılaştırılır.

    Evaluator farkları:
    short_answer:
        Cevabın içinde doğru kısa cevabın bulunması yeterlidir.

        Örnek:
            reference  = "Nigar"
            prediction = "Nigar is the shortest."
            -> correct = 1

        Kullanıldığı task'lar:
            - factual_knowledge
            - reasoning

    semantic_answer:
        Cevabın birebir aynı olması gerekmez; normalize edilmiş cevap
        içindeki anlamlı cevap ifadesinin eşleşmesini kontrol eder.

        Örnek:
            reference  = "because of heavy rain"
            prediction = "The event was cancelled because of heavy rain."
            -> correct = 1

        Kullanıldığı task:
            - linguistic_understanding

    exact_match:
        Normalize edildikten sonra prediction ile reference_answer'ın
        tamamen aynı olması gerekir.

        Örnek:
            reference  = "A B C"
            prediction = "A B C"
            -> correct = 1

            prediction = "The answer is A B C"
            -> correct = 0

        Kullanıldığı task:
            - instruction_following

    unanswerable:
        Capability evaluation yapılmaz ve None döndürülür.
        Bu task reliability/abstention evaluator tarafından değerlendirilir.

    Akış:
    record.task
        ↓
    uygun evaluator seçilir
        ↓
    score hesaplanır (0 veya 1)
        ↓
    TaskAwareCapabilityResult oluşturulur

    Örnek:
    record.task = "reasoning"
    record.prediction = "42 is the answer."
    record.reference_answer = "42"

    evaluator = "short_answer"
    score = 1

    Sonuç:
        evaluator = "short_answer"
        correct = 1

    Desteklenmeyen bir task verilirse ValueError oluşturur.
    """

    if record.task in {
        "factual_knowledge",
        "reasoning",
    }:
        if is_binary_reference(
            record.reference_answer
        ):
            evaluator = "binary_answer"
            score = binary_answer_match_score(
                record,
                adjudication_decisions=(
                    binary_adjudication_decisions
                ),
            )
        else:
            evaluator = "short_answer"
            score = short_answer_match_score(
                record.prediction,
                record.reference_answer,
            )

    elif record.task == "linguistic_understanding":
        category = record.metadata.get("category")

        if category == "reference_resolution":
            evaluator = "short_answer"
            score = short_answer_match_score(
                record.prediction,
                record.reference_answer,
            )

        elif category == "paraphrase_understanding":
            evaluator = "option_match"
            score = option_match_score(
                prediction=record.prediction,
                reference_answer=record.reference_answer,
                question=record.question,
            )

        elif category in SEMANTIC_ADJUDICATION_CATEGORIES:
            if semantic_adjudication_decisions is None:
                raise ValueError(
                    "Semantic adjudication decisions are required "
                    f"for item: {record.item_id}"
                )

            if record.item_id not in semantic_adjudication_decisions:
                raise ValueError(
                    "Missing semantic adjudication decision: "
                    f"{record.item_id}"
                )

            evaluator = "semantic_adjudication"
            score = semantic_adjudication_decisions[
                record.item_id
            ]

        else:
            raise ValueError(
                "Unsupported linguistic_understanding category: "
                f"{category!r}"
            )

    elif record.task == "instruction_following":
        evaluator = "instruction_following"

        score = instruction_following_score(
            record.prediction,
            record.reference_answer,
            category=record.metadata.get("category"),
        )

    elif record.task == "unanswerable":
        return None

    else:
        raise ValueError(
            f"Unsupported capability task: '{record.task}'"
        )

    return TaskAwareCapabilityResult(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        prediction=record.prediction,
        reference_answer=record.reference_answer,
        evaluator=evaluator,
        correct=score,
        metadata=dict(record.metadata)
    )


def evaluate_task_aware_capability(
    records: list[PredictionRecord],
    semantic_adjudication_decisions: dict[str, int] | None = None,
    binary_adjudication_decisions: dict[str, int] | None = None,
) -> list[TaskAwareCapabilityResult]:
    """Prediction listesini task-aware capability evaluator ile değerlendirir.

    Unanswerable kayıtlar capability scoring dışında tutulduğu için
    sonuç listesine eklenmez.
    """

    results: list[TaskAwareCapabilityResult] = []

    for record in records:
        result = evaluate_task_aware_prediction(
            record,
            semantic_adjudication_decisions=(
                semantic_adjudication_decisions
            ),
            binary_adjudication_decisions=(
                binary_adjudication_decisions
            ),
        )

        if result is not None:
            results.append(
                result
            )

    return results



def calculate_task_aware_accuracy(
    results: list[TaskAwareCapabilityResult]
) -> float:
    """Task-aware capability sonuçlarından overall accuracy hesaplar."""

    if not results:
        raise ValueError(
            "Cannot calculate accuracy from empty task-aware results."
        )

    correct = sum(
        result.correct
        for result in results
    )

    accuracy = correct / len(results)

    return accuracy


def summarize_task_aware_capability(
    results: list[TaskAwareCapabilityResult]
) -> dict[str, Any]:
    """Task-aware capability sonuçlarının temel özetini oluşturur."""

    if not results:
        raise ValueError(
            "Cannot summarize empty task-aware results."
        )

    correct = sum(
        result.correct
        for result in results
    )

    total = len(results)

    by_language: dict[str,dict[str, Any]] = {}
    by_task: dict[str, dict[str, Any]] = {}

    for result in results:
        language_summary = by_language.setdefault(
            result.language,
            {
                'total': 0,
                'correct': 0
            }
        )

        language_summary['total'] += 1
        language_summary['correct'] += result.correct

        task_summary = by_task.setdefault(
            result.task,
            {
                'total': 0,
                'correct': 0
            }
        )

        task_summary['total'] += 1
        task_summary['correct'] += result.correct

    for summary in by_language.values():
        summary['incorrect'] = (
            summary['total'] 
            - summary['correct']
        )
        summary['accuracy'] = (
            summary['correct']
            / summary['total']
        )

    for summary in by_task.values():
        summary['incorrect'] = (
            summary['total']
            - summary['correct']
        )
        summary['accuracy'] = (
            summary['correct']
            / summary['total']
        )



    return {
        "overall": {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total
        },
        "by_language": by_language,
        "by_task": by_task,
        "language_gap": calculate_task_aware_language_gap(
            results
        )
    }



def calculate_task_aware_language_gap(
    results: list[TaskAwareCapabilityResult],
    source_language: str = 'en',
    target_language: str = 'az'
) -> dict[str, float | None]:
    """Task-aware capability için source-target language gap hesaplar."""

    source_results = [
        result
        for result in results
        if result.language == source_language
    ]

    target_results = [
        result 
        for result in results
        if result.language == target_language
    ]

    source_accuracy = (
        calculate_task_aware_accuracy(
            source_results
        )
        if source_results
        else None
    )

    target_accuracy = (
        calculate_task_aware_accuracy(
            target_results
        )
        if target_results
        else None
    )

    absolute_gap = (
        source_accuracy - target_accuracy
        if (
            source_accuracy is not None
            and target_accuracy is not None
        )
        else None
    )

    return {
        "source_accuracy": source_accuracy,
        "target_accuracy": target_accuracy,
        "absolute_gap": absolute_gap
    }



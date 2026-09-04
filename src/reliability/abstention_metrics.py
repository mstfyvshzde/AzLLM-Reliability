"""Abstention davranışını benchmark beklentisine göre değerlendirir.

Bu modül modelin yalnızca abstain edip etmediğini değil, abstention
davranışının benchmark item için uygun olup olmadığını değerlendirir.

Amaç reliability açısından şu dört temel durumu ayırmaktır:

    correct_answer:
        Item answerable ve model cevap vermiştir.

    under_answering:
        Item answerable olduğu halde model abstain etmiştir.

    correct_abstention:
        Item unanswerable ve model doğru biçimde abstain etmiştir.

    over_answering:
        Item unanswerable olduğu halde model cevap üretmiştir.

Bu ayrım model reliability değerlendirmesinde önemlidir çünkü yüksek
capability tek başına güvenilir davranış anlamına gelmez.
"""


from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.run_inference import PredictionRecord
from src.reliability.response_status import (
    ABSTAINED_STATUS,
    ANSWERED_STATUS,
    EMPTY_STATUS,
    classify_response_status
)

# Soru cevaplanabilir ve model doğru şekilde cevap vermiş.
CORRECT_ANSWER = "correct_answer"
# Soru cevaplanabilir ama model gereksiz yere abstain etmiş
UNDER_ANSWERING = "under_answering"
# Soru cevaplanamaz ve model doğru şekilde abstain etmiş.
CORRECT_ABSTENTION = "correct_abstention"
# Soru cevaplanamaz ama model yine de cevap uydurmuş/vermiş.
OVER_ANSWERING = "over_answering"
# Model hiçbir cevap üretmemiş.
EMPTY_RESPONSE = "empty_response"


VALID_ABSTENTION_OUTCOMES = {
    CORRECT_ANSWER,
    UNDER_ANSWERING,
    CORRECT_ABSTENTION,
    OVER_ANSWERING,
    EMPTY_RESPONSE
}


@dataclass(frozen=True)
class AbstentionResult:
    """Tek bir prediction için abstention reliability sonucunu temsil eder.

    Alanlar:
        item_id:
            Benchmark item kimliği.

        pair_id:
            Semantic EN-AZ pair kimliği.

        language:
            Prediction dili.

        task:
            Benchmark task ailesi.

        prediction:
            Modelin original cevabı.

        is_answerable:
            Benchmark item'in cevaplanabilir olup olmadığını belirtir.

        response_status:
            Model cevabının answered, abstained veya empty durumu.

        outcome:
            Abstention davranışının reliability sonucu.

        metadata:
            Prediction kaydından korunan benchmark metadata alanları.
    """

    item_id: str
    pair_id: str
    language: str
    task: str
    prediction: str
    is_answerable: bool
    response_status: str
    outcome: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """AbstentionResult nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)



def classify_abstention_outcome(
    is_answerable: bool,
    response_status: str
) -> str:
    """Answerability ve response status üzerinden reliability sonucu üretir.

    Kurallar:

        answerable + answered
            -> correct_answer

        answerable + abstained
            -> under_answering

        unanswerable + abstained
            -> correct_abstention

        unanswerable + answered
            -> over_answering

        herhangi bir item + empty
            -> empty_response

    response_status geçersizse ValueError oluşturur.
    """


    valid_statuses = {
        ANSWERED_STATUS,
        ABSTAINED_STATUS,
        EMPTY_STATUS
    }

    if response_status not in valid_statuses:
        raise ValueError(
            f"Unknown response status: '{response_status}'"
        )

    if response_status == EMPTY_STATUS:
        return EMPTY_RESPONSE

    if is_answerable:
        if response_status == ANSWERED_STATUS:
            return CORRECT_ANSWER

        return UNDER_ANSWERING

    if response_status == ABSTAINED_STATUS:
        return CORRECT_ABSTENTION

    return OVER_ANSWERING


def get_answerability(
    record: PredictionRecord
) -> bool:
    """PredictionRecord metadata içinden item'in cevaplanabilir olup olmadığını okur.

    `metadata`, benchmark kaydıyla ilgili ek bilgileri tutan sözlüktür.

    Örnek metadata:
    {
        "category": "factual_qa",
        "difficulty": "medium",
        "is_answerable": True
    }

    Bu fonksiyon `is_answerable` değerini alır:

    True  → soru cevaplanabilir
    False → soru cevaplanamaz

    `is_answerable` alanı yoksa KeyError,
    değer bool değilse TypeError oluşturur.
    """

    if 'is_answerable' not in record.metadata:
        raise KeyError(
            f"Missing 'is_answerable' metadata for item '{record.item_id}'."
        )

    is_answerable = record.metadata['is_answerable']    

    if not isinstance(is_answerable, bool):
        raise TypeError(
            f"'is_answerable' must be bool for item '{record.item_id}'."
        )

    return is_answerable



def create_abstention_result(
    record: PredictionRecord
) -> AbstentionResult:
    """Tek bir PredictionRecord için abstention reliability sonucunu oluşturur.

    Önce benchmark item'inin cevaplanabilir olup olmadığını metadata içinden alır,
    sonra model cevabını `answered`, `abstained` veya `empty` olarak sınıflandırır.
    Bu iki bilgiyi birleştirerek final reliability outcome üretir.

    Örnek:
    is_answerable = False
    prediction = "I don't know."

    response_status = "abstained"
    outcome = "correct_abstention"

    Sonuç:
    AbstentionResult(
        ...,
        is_answerable=False,
        response_status="abstained",
        outcome="correct_abstention"
    )
    """

    is_answerable = get_answerability(record)

    response_status = classify_response_status(record.prediction)

    outcome = classify_abstention_outcome(
        is_answerable=is_answerable,
        response_status=response_status
    )


    return AbstentionResult(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        prediction=record.prediction,
        is_answerable=is_answerable,
        response_status=response_status,
        outcome=outcome,
        metadata=dict(record.metadata),
    )



def evaluate_abstention(
    records: list[PredictionRecord]
) -> list[AbstentionResult]:
    """Tüm PredictionRecord kayıtları için abstention reliability evaluation çalıştırır.

    Her kayıt `create_abstention_result()` fonksiyonuna gönderilir ve
    bir AbstentionResult oluşturulur.

    Örnek:
    records:
        answerable=True,  prediction="Paris"
        answerable=False, prediction="I don't know."

    Sonuç:
    [
        AbstentionResult(outcome="correct_answer", ...),
        AbstentionResult(outcome="correct_abstention", ...)
    ]

    Input sırası korunur. Boş liste verilirse boş liste döndürülür.
    """

    return [
        create_abstention_result(
            record
        )
        for record in records
    ]



def summarize_abstention_results(
    results: list[AbstentionResult]
) -> dict[str, Any]:
    """Abstention reliability sonuçlarının aggregate özetini oluşturur.

    Döndürülen temel alanlar:
        total:
            Toplam evaluated item sayısı.

        correct_answer:
            Answerable item üzerinde cevap verilen kayıt sayısı.

        under_answering:
            Answerable item üzerinde abstain edilen kayıt sayısı.

        correct_abstention:
            Unanswerable item üzerinde doğru abstention sayısı.

        over_answering:
            Unanswerable item üzerinde cevap verilen kayıt sayısı.

        empty_response:
            Boş response sayısı.

        abstention_accuracy:
            Doğru answer/abstention kararlarının toplam item sayısına oranı.

        over_answering_rate:
            Unanswerable item'larda modelin cevap verme oranı.

        under_answering_rate:
            Answerable item'larda modelin gereksiz abstain etme oranı.
    """

    if not results:
        raise ValueError(
            "Cannot summarize empty abstention results."
        )

    counts = {
        CORRECT_ANSWER: 0,
        UNDER_ANSWERING: 0,
        CORRECT_ABSTENTION: 0,
        OVER_ANSWERING: 0,
        EMPTY_RESPONSE: 0,
    }

    for result in results:
        if result.outcome not in VALID_ABSTENTION_OUTCOMES:
            raise ValueError(
                f"Unknown abstention outcome: '{result.outcome}'"
            )

        counts[result.outcome] += 1

    total = len(results)

    correct_decisions = (
        counts[CORRECT_ANSWER]
        + counts[CORRECT_ABSTENTION]
    )

    answerable_total = sum(
        1 
        for result in results
        if result.is_answerable
    )

    unanswerable_total = (
        total - answerable_total
    )

    over_answering_rate = (
        counts[OVER_ANSWERING] 
        / unanswerable_total
        if unanswerable_total > 0
        else 0.0
    )

    under_answering_rate = (
        counts[UNDER_ANSWERING]
        / answerable_total
        if answerable_total > 0
        else 0.0
    )


    return {
        "total": total,
        "correct_answer": counts[CORRECT_ANSWER],
        "under_answering": counts[UNDER_ANSWERING],
        "correct_abstention": counts[CORRECT_ABSTENTION],
        "over_answering": counts[OVER_ANSWERING],
        "empty_response": counts[EMPTY_RESPONSE],
        # decision_accuracy yalnızca modelin doğru zamanda
        # answer / abstain kararı verip vermediğini ölçer.
        #
        # Bu değer answerable item üzerindeki cevabın içerik olarak
        # doğru olduğunu iddia etmez; capability correctness ayrı
        # task-aware evaluator tarafından ölçülür.
        "decision_accuracy": (
            correct_decisions
            / total
        ),

        # Backward-compatible diagnostic alias.
        # Final research reporting için decision_accuracy kullanılmalıdır.
        "abstention_accuracy": (
            correct_decisions
            / total
        ),
        "over_answering_rate": over_answering_rate,
        "under_answering_rate": under_answering_rate
    }
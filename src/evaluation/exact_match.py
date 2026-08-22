"""Normalized exact-match capability metriğini hesaplar.

Bu modül model prediction değerlerini reference answer değerleriyle normalize
edilmiş biçimde karşılaştırır.

Exact match özellikle kısa ve deterministik cevap beklenen benchmark görevleri
için temel capability metriği olarak kullanılır.

Örnek:
    prediction = "  Four. "
    reference_answer = "four"

Normalization sonrası:
    prediction = "four"
    reference_answer = "four"

Sonuç:
    exact_match = 1

Item-level sonuçların korunması daha sonraki language-gap, task-level ve
failure analysis aşamalarında aynı evaluation artifact'ının kullanılmasını
sağlar.
"""


from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.normalize_answer import normalize_answer
from src.evaluation.run_inference import PredictionRecord

@dataclass(frozen=True)
class ExactMatchResult:
    """Tek bir prediction için exact-match evaluation sonucunu temsil eder.

    Alanlar:
        item_id:
            Benchmark item kimliği.

        pair_id:
            Semantik EN-AZ pair kimliği.

        language:
            Prediction kaydının dili.

        task:
            Benchmark task ailesi.

        prediction:
            Modelin original cevabı.

        reference_answer:
            Benchmark reference answer değeri.

        normalized_prediction:
            Evaluation öncesinde normalize edilmiş model cevabı.

        normalized_reference:
            Normalize edilmiş reference answer.

        exact_match:
            Cevaplar eşleşiyorsa 1, aksi durumda 0.

        metadata:
            Prediction kaydından korunan benchmark metadata alanları.
    """

    item_id: str
    pair_id: str
    language: str
    task: str
    prediction: str
    reference_answer: str
    normalized_prediction: str
    normalized_reference: str
    exact_match: int
    metadata: dict[str, Any]


    def to_dict(self) -> dict[str, Any]:
        """ExactMatchResult nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)


def exact_match_score(
    prediction: str,
    reference_answer: str
) -> int:
    """Prediction ve reference answer için normalized exact match hesaplar.

    İki cevap normalize edildikten sonra tamamen aynıysa 1, farklıysa 0
    döndürür.

    Örnek:
        prediction = "Four."
        reference_answer = "four"

        → 1

        prediction = "five"
        reference_answer = "four"

        → 0
    """

    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference_answer)

    return int(normalized_prediction == normalized_reference)


def evaluate_prediction(
    record: PredictionRecord
) -> ExactMatchResult:
    """Tek bir PredictionRecord için item-level exact-match sonucu üretir.

    Original prediction ve reference answer korunurken normalized biçimleri
    ayrıca kaydedilir.

    Bu ayrım, evaluation sonucunun daha sonra denetlenebilmesini ve normalization
    davranışının hangi cevapları eşit kabul ettiğinin açıkça görülebilmesini
    sağlar.
    """

    normalized_prediction = normalize_answer(record.prediction)
    normalized_reference = normalize_answer(record.reference_answer)

    score = int(normalized_prediction == normalized_reference)


    return ExactMatchResult(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        prediction=record.prediction,
        reference_answer=record.reference_answer,
        normalized_prediction=normalized_prediction,
        normalized_reference=normalized_reference,
        exact_match=score,
        metadata=dict(record.metadata)
    )



def evaluate_exact_match(
    records: list[PredictionRecord]
) -> list[ExactMatchResult]:
    """Tüm prediction kayıtları için item-level exact-match evaluation çalıştırır.

    Her PredictionRecord, `evaluate_prediction()` fonksiyonuna gönderilir ve
    karşılığında bir ExactMatchResult oluşturulur.

    Örnek input:
    records = [
        PredictionRecord(prediction="Four.", reference_answer="four", ...),
        PredictionRecord(prediction="five", reference_answer="four", ...),
    ]

    Örnek sonuç:
    [
        ExactMatchResult(exact_match=1, ...),
        ExactMatchResult(exact_match=0, ...),
    ]

    Kayıtların sırası korunur.
    Boş bir prediction listesi verilirse boş liste döndürülür.
    """

    return [
        evaluate_prediction(record)
        for record in records
    ]


def calculate_accuracy(
    results: list[ExactMatchResult]
) -> float:
    """Exact-match sonuçlarından accuracy değerini hesaplar.

    Accuracy:

        doğru prediction sayısı
        -----------------------
        toplam prediction sayısı

    Sonuç 0.0 ile 1.0 arasında float olarak döndürülür.

    Boş result listesi için accuracy tanımsız olduğundan ValueError oluşturur.
    """

    if not results:
        raise ValueError(
            "Cannot calculate accuracy from empty results."
        )

    correct_count = sum(
        result.exact_match
        for result in results
    )

    accuracy = correct_count / len(results)

    return accuracy


def summarize_exact_match(
    results: list[ExactMatchResult]
) -> dict[str, Any]:
    """Exact-match sonuçlarının temel aggregate özetini oluşturur.

    Döndürülen özet:
        total:
            Toplam evaluated item sayısı.

        correct:
            Exact-match doğru item sayısı.

        incorrect:
            Exact-match yanlış item sayısı.

        accuracy:
            Overall exact-match accuracy.

    Bu fonksiyon henüz EN-AZ gap veya task-level istatistik hesaplamaz.
    Bu analizler ayrı evaluation/analysis katmanlarında yapılacaktır.
    """

    if not results: 
        raise ValueError(
            "Cannot summarize empty exact-match results."
        )

    correct = sum(
        result.exact_match
        for result in results
    )

    total = len(results)


    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total
    }
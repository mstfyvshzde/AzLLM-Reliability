"""Kısa ve deterministik cevaplar için answer-aware capability evaluation yapar.

Bu modül strict exact-match metriğini değiştirmez.

Amaç, kısa reference answer'ın prediction içinde açık ve güvenli biçimde
ifade edilip edilmediğini ölçmektir.

Örnekler:

    reference = "Nigar"
    prediction = "Nigar is the shortest."
    -> doğru

Azerbaycanca gibi eklemeli dillerde reference answer prediction içinde
çekimli biçimde bulunabilir:

    reference = "Nil"
    prediction = "Çay Nildir."
    -> doğru

    reference = "Mina"
    prediction = "Minanın daha çox qələmi var."
    -> doğru

Bu nedenle evaluator:

1. normalize edilmiş exact match'i,
2. bağımsız token eşleşmesini,
3. kontrollü Azerbaycanca ek eşleşmesini

destekler.

Serbest substring matching kullanılmaz. Böylece örneğin:

    reference = "no"
    prediction = "nobody"

yanlış biçimde eşleşmez.

Instruction-following ve unanswerable görevleri ayrı evaluator'lar tarafından
değerlendirilmelidir.
"""

from __future__ import annotations

import re

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.normalize_answer import normalize_answer
from src.evaluation.run_inference import PredictionRecord


@dataclass(frozen=True)
class ShortAnswerMatchResult:
    """Tek bir short-answer prediction değerlendirme sonucunu temsil eder."""

    item_id: str
    pair_id: str
    language: str
    task: str
    prediction: str
    reference_answer: str
    normalized_prediction: str
    normalized_reference: str
    short_answer_match: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Result nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)


# Sayıları, kesirleri ve EN/AZ alfabetik token'ları korur.
#
# Örnek:
#
#     "Paris'tir" -> ["paris", "tir"]
#     "50-dir"    -> ["50", "dir"]
#     "1/2-dir"   -> ["1/2", "dir"]
#
_MATCH_TOKEN_PATTERN = re.compile(
    r"\d+(?:/\d+)?|[a-zəğıöşüç]+",
    flags=re.IGNORECASE,
)


# Reference token'ın prediction içinde çekimli biçimde görünmesi durumunda
# kabul edilebilecek kontrollü Azerbaycanca son ekler.
#
# Bu liste bilinçli olarak sınırlıdır.
# Amaç genel bir morphological analyzer yazmak değil,
# benchmark scoring sırasında açık çekim eklerinden kaynaklanan
# false-negative'leri azaltmaktır.
_AZ_ALLOWED_SUFFIX_TAILS = frozenset(
    {
        # Bağlayıcı / relational biçimler.
        "n",

        # Yalın isim çekimleri.
        "a",
        "ə",
        "ya",
        "yə",
        "ı",
        "i",
        "u",
        "ü",
        "nı",
        "ni",
        "nu",
        "nü",

        # Yönelme / bulunma / ayrılma.
        "da",
        "də",
        "nda",
        "ndə",
        "dan",
        "dən",
        "ndan",
        "ndən",

        # İlgi / iyelik benzeri biçimler.
        "ın",
        "in",
        "un",
        "ün",
        "nın",
        "nin",
        "nun",
        "nün",

        # Birliktelik.
        "la",
        "lə",
        "yla",
        "ylə",

        # Çoğul.
        "lar",
        "lər",

        # Copula.
        "dır",
        "dir",
        "dur",
        "dür",
        "tır",
        "tir",
        "tur",
        "tür",

        # Yaygın çoğul + çekim kombinasyonları.
        "ların",
        "lərin",
        "larda",
        "lərdə",
        "lardan",
        "lərdən",
        "ları",
        "ləri",

        # Yaygın iyelik + hal kombinasyonları.
        "ında",
        "ində",
        "unda",
        "ündə",
        "ından",
        "indən",
        "undan",
        "ündən",
    }
)


def _match_tokens(text: str) -> list[str]:
    """Metni güvenli answer-matching token'larına ayırır."""

    normalized = normalize_answer(text)

    return [
        match.group(0)
        for match in _MATCH_TOKEN_PATTERN.finditer(normalized)
    ]


def _is_az_inflected_match(
    reference_token: str,
    prediction_token: str,
) -> bool:
    """Prediction token'ı reference'ın kontrollü AZ çekimli biçimi mi kontrol eder."""

    if reference_token == prediction_token:
        return True

    # Çok kısa alfabetik token'larda prefix matching risklidir.
    #
    # Sayılar zaten tokenization aşamasında bağımsız ele alınır.
    if len(reference_token) < 3:
        return False

    if not prediction_token.startswith(reference_token):
        return False

    suffix = prediction_token[len(reference_token):]

    if not suffix:
        return False

    return suffix in _AZ_ALLOWED_SUFFIX_TAILS


def _contains_reference_token_sequence(
    prediction: str,
    reference_answer: str,
) -> bool:
    """Reference token dizisinin prediction içinde güvenli biçimde bulunmasını kontrol eder."""

    prediction_tokens = _match_tokens(prediction)
    reference_tokens = _match_tokens(reference_answer)

    if not reference_tokens:
        return False

    reference_length = len(reference_tokens)

    if len(prediction_tokens) < reference_length:
        return False

    for index in range(
        (len(prediction_tokens) - reference_length) + 1
    ):
        candidate = prediction_tokens[
            index:index + reference_length
        ]

        token_matches = all(
            _is_az_inflected_match(
                reference_token=reference_token,
                prediction_token=prediction_token,
            )
            for reference_token, prediction_token in zip(
                reference_tokens,
                candidate,
                strict=True,
            )
        )

        if token_matches:
            return True

    return False


def contains_contradictory_answer(
    prediction: str,
    reference_answer: str,
) -> bool:
    """Prediction reference cevabı verip sonra açıkça tersine çeviriyor mu kontrol eder."""

    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference_answer)

    if not normalized_reference:
        return False

    patterns = (
        rf"\b{re.escape(normalized_reference)}\b.*\bbut\b",
        rf"\b{re.escape(normalized_reference)}\b.*\bhowever\b",
        rf"\b{re.escape(normalized_reference)}\b.*\bactually\b",
    )

    return any(
        re.search(
            pattern,
            normalized_prediction,
            flags=re.DOTALL,
        )
        for pattern in patterns
    )


def short_answer_match_score(
    prediction: str,
    reference_answer: str,
) -> int:
    """Prediction içinde reference answer'ın güvenli biçimde bulunmasını ölçer.

    Değerlendirme sırası:

    1. Normalize edilmiş exact match.
    2. Açık contradiction kontrolü.
    3. EN/AZ token-aware containment.
    4. Kontrollü Azerbaycanca çekim eşleşmesi.

    Serbest substring matching yapılmaz.
    """

    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference_answer)

    if normalized_prediction == normalized_reference:
        return 1

    if not normalized_reference:
        return 0

    if contains_contradictory_answer(
        prediction,
        reference_answer,
    ):
        return 0

    if _contains_reference_token_sequence(
        prediction,
        reference_answer,
    ):
        return 1

    return 0


def evaluate_short_answer_prediction(
    record: PredictionRecord,
) -> ShortAnswerMatchResult:
    """Tek bir PredictionRecord için short-answer match sonucu üretir."""

    normalized_prediction = normalize_answer(
        record.prediction
    )

    normalized_reference = normalize_answer(
        record.reference_answer
    )

    score = short_answer_match_score(
        prediction=record.prediction,
        reference_answer=record.reference_answer,
    )

    return ShortAnswerMatchResult(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        prediction=record.prediction,
        reference_answer=record.reference_answer,
        normalized_prediction=normalized_prediction,
        normalized_reference=normalized_reference,
        short_answer_match=score,
        metadata=dict(record.metadata),
    )


def evaluate_short_answer_matches(
    records: list[PredictionRecord],
) -> list[ShortAnswerMatchResult]:
    """Prediction listesini short-answer evaluator ile değerlendirir."""

    return [
        evaluate_short_answer_prediction(record)
        for record in records
    ]


def calculate_short_answer_accuracy(
    results: list[ShortAnswerMatchResult],
) -> float:
    """Short-answer sonuçlarından accuracy hesaplar."""

    if not results:
        raise ValueError(
            "Cannot calculate accuracy from empty short-answer results."
        )

    correct = sum(
        result.short_answer_match
        for result in results
    )

    accuracy = correct / len(results)

    return accuracy


def summarize_short_answer_matches(
    results: list[ShortAnswerMatchResult],
) -> dict[str, Any]:
    """Short-answer evaluation sonuçlarının temel özetini oluşturur."""

    if not results:
        raise ValueError(
            "Cannot summarize empty short-answer results."
        )

    correct = sum(
        result.short_answer_match
        for result in results
    )

    total = len(results)

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
    }


SHORT_ANSWER_TASKS = {
    "factual_knowledge",
    "reasoning",
}


def is_short_answer_task(
    task: str,
) -> bool:
    """Task'in short-answer evaluator kullanıp kullanmayacağını belirler."""

    return task in SHORT_ANSWER_TASKS


def filter_short_answer_records(
    records: list[PredictionRecord],
) -> list[PredictionRecord]:
    """Yalnızca short-answer task kayıtlarını döndürür."""

    return [
        record
        for record in records
        if is_short_answer_task(record.task)
    ]

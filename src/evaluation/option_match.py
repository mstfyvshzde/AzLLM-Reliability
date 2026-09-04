"""Explicit candidate-option içeren görevler için deterministik evaluator sağlar.

Bu evaluator özellikle paraphrase_understanding gibi,
prompt içinde iki açık seçenek verilen görevlerde kullanılır.

Amaç yalnızca reference cevabın prediction içinde bulunmasını kontrol etmek değil,
yanlış competing option'ın da aynı anda seçilmediğinden emin olmaktır.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import re

from src.evaluation.normalize_answer import normalize_answer
from src.evaluation.run_inference import PredictionRecord


@dataclass(frozen=True)
class OptionMatchResult:
    """Tek bir option-based prediction evaluation sonucunu temsil eder."""

    item_id: str
    pair_id: str
    language: str
    task: str
    prediction: str
    reference_answer: str
    option_match: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Result nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)


def extract_explicit_options(
    question: str,
) -> list[str]:
    """Prompt içindeki explicit iki answer option'ını çıkarır.

    Benchmark paraphrase soruları şu yapıyı kullanır:

        Answer: 'option one' or 'option two.'
        Cavab verin: 'variant one' və ya 'variant two.'

    Fonksiyon quoted seçenekleri çıkarır ve tam olarak iki seçenek
    bulunamazsa boş liste döndürür.
    """

    quoted = re.findall(
        r"'([^']+)'",
        question,
    )

    if len(quoted) < 3:
        return []

    options = quoted[-2:]

    return [
        option.strip().rstrip(".")
        for option in options
    ]


def contains_normalized_phrase(
    text: str,
    phrase: str,
) -> bool:
    """Normalize edilmiş phrase text içinde token dizisi olarak bulunuyor mu?"""

    normalized_text = normalize_answer(text)
    normalized_phrase = normalize_answer(phrase)

    if not normalized_phrase:
        return False

    text_tokens = normalized_text.split()
    phrase_tokens = normalized_phrase.split()

    length = len(phrase_tokens)

    for index in range(
        len(text_tokens) - length + 1
    ):
        if (
            text_tokens[index:index + length]
            == phrase_tokens
        ):
            return True

    return False


def option_match_score(
    prediction: str,
    reference_answer: str,
    question: str | None = None,
) -> int:
    """Prediction doğru explicit option'ı seçmiş mi kontrol eder.

    Kurallar:

    1. Reference answer prediction içinde bulunmalıdır.
    2. Prompt'tan competing option güvenle çıkarılabiliyorsa,
       prediction competing option'ı içermemelidir.
    3. Prompt parse edilemiyorsa yalnız reference containment uygulanır.

    Böylece:

        correct option only
            -> 1

        correct + competing option
            -> 0

        competing option only
            -> 0
    """

    if not contains_normalized_phrase(
        prediction,
        reference_answer,
    ):
        return 0

    if question is None:
        return 1

    options = extract_explicit_options(
        question
    )

    if len(options) != 2:
        return 1

    normalized_reference = normalize_answer(
        reference_answer
    )

    competing_options = [
        option
        for option in options
        if normalize_answer(option)
        != normalized_reference
    ]

    if len(competing_options) != 1:
        return 1

    competing = competing_options[0]

    if contains_normalized_phrase(
        prediction,
        competing,
    ):
        return 0

    return 1


def evaluate_option_prediction(
    record: PredictionRecord,
) -> OptionMatchResult:
    """Tek bir PredictionRecord için option-match sonucu üretir."""

    score = option_match_score(
        prediction=record.prediction,
        reference_answer=record.reference_answer,
        question=record.question,
    )

    return OptionMatchResult(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        prediction=record.prediction,
        reference_answer=record.reference_answer,
        option_match=score,
        metadata=dict(record.metadata),
    )


def evaluate_option_matches(
    records: list[PredictionRecord],
) -> list[OptionMatchResult]:
    """Prediction listesini option-match evaluator ile değerlendirir."""

    return [
        evaluate_option_prediction(record)
        for record in records
    ]


def calculate_option_accuracy(
    results: list[OptionMatchResult],
) -> float:
    """Option-match sonuçlarından accuracy hesaplar."""

    if not results:
        raise ValueError(
            "Cannot calculate accuracy from empty option-match results."
        )

    correct = sum(
        result.option_match
        for result in results
    )

    return correct / len(results)

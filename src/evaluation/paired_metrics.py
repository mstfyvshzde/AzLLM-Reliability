"""Paired EN-AZ exact-match sonuçlarını pair seviyesinde karşılaştırır.

Bu modül aynı semantic benchmark pair'ine ait English ve Azerbaijani
ExactMatchResult kayıtlarını eşleştirir.

Amaç yalnızca aggregate language accuracy farkını ölçmek değil, aynı
semantic görev üzerinde model davranışının diller arasında nasıl değiştiğini
pair seviyesinde analiz etmektir.

Örnek:
    EN exact_match = 1
    AZ exact_match = 0

Bu durum language-induced capability degradation olarak işaretlenebilir.
"""


from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.exact_match import ExactMatchResult

@dataclass(frozen=True)
class PairedExactMatchResult:
    """Tek bir EN-AZ pair için paired capability sonucunu temsil eder.

    Alanlar:
        pair_id:
            Semantic benchmark pair kimliği.

        task:
            Pair'in ait olduğu benchmark task ailesi.

        source_language:
            Source language kodu.

        target_language:
            Target language kodu.

        source_item_id:
            Source language benchmark item kimliği.

        target_item_id:
            Target language benchmark item kimliği.

        source_exact_match:
            Source language prediction doğruysa 1, yanlışsa 0.

        target_exact_match:
            Target language prediction doğruysa 1, yanlışsa 0.

        transition:
            Pair-level capability transition sınıfı.

        metadata:
            Pair analizinde kullanılacak benchmark metadata alanları.
    """

    pair_id: str
    task: str
    source_language: str
    target_language: str
    source_item_id: str
    target_item_id: str
    source_exact_match: int
    target_exact_match: int
    transition: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """PairedExactMatchResult nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)


def classify_transition(
    source_exact_match: int,
    target_exact_match: int
) -> str:
    """Source-target exact-match durumunu pair-level transition olarak sınıflandırır.

    Olası sınıflar:

        both_correct:
            Source ve target cevaplarının ikisi de doğru.

        source_only_correct:
            Source doğru, target yanlış.
            Language-induced degradation için temel sinyaldir.

        target_only_correct:
            Source yanlış, target doğru.

        both_incorrect:
            Her iki dilde de cevap yanlış.

    exact_match değerleri yalnızca 0 veya 1 olabilir.
    """

    valid_scores = {0, 1}

    if source_exact_match not in valid_scores:
        raise ValueError(
            f"Invalid source exact-match value: {source_exact_match}"
        )

    if target_exact_match not in valid_scores:
        raise ValueError(
            f"Invalid target exact-match value: {target_exact_match}"
        )

    if source_exact_match == 1 and target_exact_match ==1:
        return 'both_correct'

    if source_exact_match == 1 and target_exact_match == 0:
        return 'source_only_correct'

    if source_exact_match == 0 and target_exact_match == 1:
        return 'target_only_correct'

    return 'both_incorrect'


def group_results_by_pair(
    results: list[ExactMatchResult]
) -> dict[str, list[ExactMatchResult]]:
    """Exact-match sonuçlarını ortak pair_id değerine göre gruplar.

    Aynı semantik sorunun İngilizce ve Azerbaycanca sonuçlarını aynı listede toplar.

    Örnek:
    reasoning_001_en → pair_id="reasoning_001"
    reasoning_001_az → pair_id="reasoning_001"

    Sonuç:
    {
        "reasoning_001": [
            result_en,
            result_az
        ]
    }

    Bu gruplama daha sonra aynı soru için EN ve AZ performansını doğrudan
    karşılaştırmak için kullanılır.
    """

    grouped: dict[
        str, 
        list[ExactMatchResult]
    ] = defaultdict(list)

    for result in results:
        grouped [
            result.pair_id
        ].append(
            result
        )

    return dict(grouped)


def create_paired_result(
    pair_results: list[ExactMatchResult],
    source_language: str = 'en',
    target_language: str = 'az'
) -> PairedExactMatchResult:
    """Tek bir EN-AZ pair içindeki source ve target exact-match sonuçlarını eşleştirir.

    `source_language` varsayılan olarak İngilizce (`en`),
    `target_language` ise Azerbaycanca (`az`) olarak kullanılır.

    Örnek:
    pair_results:
        reasoning_001_en → exact_match=1
        reasoning_001_az → exact_match=0

    Sonuç:
    PairedExactMatchResult(
        pair_id="reasoning_001",
        source_exact_match=1,
        target_exact_match=0,
        transition=...
    )

    Fonksiyon ayrıca:
    - source için tam 1 kayıt var mı?
    - target için tam 1 kayıt var mı?
    - pair_id değerleri aynı mı?
    - task değerleri aynı mı?

    kontrol eder. Kurallardan biri bozulursa ValueError oluşturur.
    """

    source_results = [
        result
        for result in pair_results
        if result.language == source_language
    ]

    target_results = [
        result
        for result in pair_results
        if result.language == target_language  
    ]

    if len(source_results) != 1:
        raise ValueError(
            f"Pair must contain exactly one '{source_language}' result."
        )

    if len(target_results) != 1:
        raise ValueError(
            f"Pair must contain exactly one '{target_language}' result."
        )

    source_result = source_results[0]
    target_result = target_results[0]

    if source_result.pair_id != target_result.pair_id:
        raise ValueError(
            "Source and target results must share the same pair_id."
        )

    if source_result.task != target_result.task:
        raise ValueError(
            f"Task mismatch inside pair '{source_result.pair_id}'."
        )

    transition = classify_transition(
        source_exact_match=source_result.exact_match,
        target_exact_match=target_result.exact_match
    )

    metadata = dict(source_result.metadata)


    return PairedExactMatchResult(
        pair_id=source_result.pair_id,
        task=source_result.task,
        source_language=source_language,
        target_language=target_language,
        source_item_id=source_result.item_id,
        target_item_id=target_result.item_id,
        source_exact_match=source_result.exact_match,
        target_exact_match=target_result.exact_match,
        transition=transition,
        metadata=metadata
    )



def evaluate_paired_exact_match(
    results: list[ExactMatchResult],
    source_language: str = 'en',
    target_language: str = 'az'
) -> list[PairedExactMatchResult]:
    """Tüm exact-match sonuçlarını pair seviyesinde EN-AZ olarak karşılaştırır.

    Önce sonuçları `pair_id` değerine göre gruplar, ardından her pair için
    `create_paired_result()` kullanarak source-target karşılaştırması oluşturur.

    Örnek:
    reasoning_001_en → exact_match=1
    reasoning_001_az → exact_match=0

    reasoning_002_en → exact_match=1
    reasoning_002_az → exact_match=1

    Sonuç:
    [
        PairedExactMatchResult(
            pair_id="reasoning_001",
            source_exact_match=1,
            target_exact_match=0,
            ...
        ),
        PairedExactMatchResult(
            pair_id="reasoning_002",
            source_exact_match=1,
            target_exact_match=1,
            ...
        )
    ]

    Pair sırası, input içinde ilk görüldüğü sıraya göre korunur.
    """

    grouped_results = group_results_by_pair(results)

    return [
        create_paired_result(
            pair_results=pair_results,
            source_language=source_language,
            target_language=target_language
        )
        for pair_results in grouped_results.values()
    ]



def summarize_paired_results(
    results: list[PairedExactMatchResult]
) -> dict[str, Any]:
    """Pair-level transition sonuçlarının aggregate özetini oluşturur.

    Döndürülen alanlar:
        total_pairs:
            Toplam paired benchmark örneği.

        both_correct:
            Her iki dilde doğru pair sayısı.

        source_only_correct:
            Yalnızca source dilde doğru pair sayısı.

        target_only_correct:
            Yalnızca target dilde doğru pair sayısı.

        both_incorrect:
            Her iki dilde yanlış pair sayısı.

        degradation_rate:
            source_only_correct / total_pairs

        recovery_rate:
            target_only_correct / total_pairs

        consistency_rate:
            Aynı correctness davranışını gösteren pair oranı.
            both_correct + both_incorrect / total_pairs
    """

    if not results:
        raise ValueError(
            "Cannot summarize empty paired results."
        )

    transition_counts = {
        "both_correct": 0,
        "source_only_correct": 0,
        "target_only_correct": 0,
        "both_incorrect": 0
    }

    for result in results:
        if result.transition not in transition_counts:
            raise ValueError(
                f"Unknown paired transition: '{result.transition}'"
            )

        transition_counts[
            result.transition
        ] += 1

    total_pairs = len(results)



    consistent_pairs = (
        transition_counts["both_correct"]
        + transition_counts["both_incorrect"]
    )

    return {
        "total_pairs": total_pairs,
        **transition_counts,
        "degradation_rate": (
            transition_counts["source_only_correct"]
            / total_pairs
        ),
        "recovery_rate": (
            transition_counts["target_only_correct"]
            / total_pairs
        ),
        "consistency_rate": (
            consistent_pairs
            / total_pairs
        )
    }
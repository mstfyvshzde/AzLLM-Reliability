"""Paired EN-AZ reliability sonuçlarını pair seviyesinde karşılaştırır.

Bu modül aynı semantic benchmark pair'ine ait English ve Azerbaijani
AbstentionResult kayıtlarını eşleştirir.

Amaç modelin reliability davranışının diller arasında nasıl değiştiğini
pair seviyesinde analiz etmektir.

Örnek:
    EN outcome = correct_answer
    AZ outcome = under_answering

Bu durum source language'da güvenilir davranış korunurken target language'da
reliability degradation oluştuğunu gösterebilir.
"""


from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any

from src.reliability.abstention_metrics import (
    CORRECT_ABSTENTION,
    CORRECT_ANSWER,
    EMPTY_RESPONSE,
    OVER_ANSWERING,
    UNDER_ANSWERING,
    AbstentionResult,
)


RELIABLE_OUTCOMES = {
    CORRECT_ANSWER,
    CORRECT_ABSTENTION,
}

UNRELIABLE_OUTCOMES = {
    UNDER_ANSWERING,
    OVER_ANSWERING,
    EMPTY_RESPONSE,
}


@dataclass(frozen=True)
class PairedReliabilityResult:
    """Tek bir EN-AZ pair için paired reliability sonucunu temsil eder.

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
            Source language item kimliği.

        target_item_id:
            Target language item kimliği.

        source_outcome:
            Source language abstention outcome değeri.

        target_outcome:
            Target language abstention outcome değeri.

        source_reliable:
            Source reliability kararı doğruysa True.

        target_reliable:
            Target reliability kararı doğruysa True.

        transition:
            Pair-level reliability transition sınıfı.

        metadata:
            Pair analizinde kullanılacak benchmark metadata alanları.
    """

    pair_id: str
    task: str
    source_language: str
    target_language: str
    source_item_id: str
    target_item_id: str
    source_outcome: str
    target_outcome: str
    source_reliable: bool
    target_reliable: bool
    transition: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """PairedReliabilityResult nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)



def is_reliable_outcome(
    outcome: str
) -> bool:
    """Abstention outcome değerinin reliability açısından doğru davranış olup olmadığını belirler.

    Reliable outcome:
    correct_answer
    correct_abstention

    Unreliable outcome:
    under_answering
    over_answering
    empty_response

    Örnek:
    outcome = "correct_answer"
    → True

    outcome = "over_answering"
    → False

    Bilinmeyen bir outcome verilirse ValueError oluşturur.
    """

    if outcome in RELIABLE_OUTCOMES:
        return True

    if outcome in UNRELIABLE_OUTCOMES:
        return False

    raise ValueError(
        f"Unknown abstention outcome: '{outcome}'"
    )



def classify_reliability_transition(
    source_reliable: bool,
    target_reliable: bool
) -> str:
    """Source-target reliability durumunu pair-level transition olarak sınıflandırır.

    Olası sınıflar:

        both_reliable:
            Her iki dilde de reliability kararı doğru.

        source_only_reliable:
            Source doğru, target reliability hatası yapmış.

        target_only_reliable:
            Source reliability hatası yapmış, target doğru.

        both_unreliable:
            Her iki dilde de reliability hatası oluşmuş.
    """

    if source_reliable and target_reliable:
        return 'both_reliable'

    if source_reliable and not target_reliable:
        return 'source_only_reliable'

    if not source_reliable and target_reliable:
        return 'target_only_reliable'

    return 'both_unreliable'


def group_reliability_by_pair(
    results: list[AbstentionResult]
) -> dict[str, list[AbstentionResult]]:
    """AbstentionResult kayıtlarını pair_id değerine göre gruplar.

    `grouped`, her pair_id için bir AbstentionResult listesi tutan sözlüktür.

    Örnek:
    grouped = {
        "reasoning_001": [],
        "reasoning_002": []
    }

    `defaultdict(list)` sayesinde yeni bir pair_id görüldüğünde boş liste
    otomatik oluşturulur.

    Sonra her result kendi pair_id grubuna eklenir:

    for result in results:
        grouped[result.pair_id].append(result)

    Örnek:
    reasoning_001_en → pair_id="reasoning_001"
    reasoning_001_az → pair_id="reasoning_001"

    Sonuç:
    {
        "reasoning_001": [
            reasoning_001_en,
            reasoning_001_az
        ]
    }
    """

    grouped: dict[
        str,
        list[AbstentionResult]
    ] = defaultdict(list)

    for result in results:
        grouped[
            result.pair_id
        ].append(result)

    return dict(grouped)



def create_paired_reliability_result(
    pair_results: list[AbstentionResult],
    source_language: str = 'en',
    target_language: str = 'az'
) -> PairedReliabilityResult:
    """Tek bir EN-AZ pair içindeki source ve target reliability sonuçlarını eşleştirir.

    Örnek:
    pair_results:
        reasoning_001_en → outcome="correct_answer"
        reasoning_001_az → outcome="under_answering"

    Fonksiyon:
    - source language için tam 1 kayıt var mı?
    - target language için tam 1 kayıt var mı?
    - pair_id aynı mı?
    - task aynı mı?
    - is_answerable değeri aynı mı?

    kontrol eder.

    Sonra:
    correct_answer / correct_abstention → reliable=True
    under_answering / over_answering / empty_response → reliable=False

    Örnek sonuç:
    source_reliable = True
    target_reliable = False
    transition = "source_only_reliable"

    Son olarak bu bilgileri PairedReliabilityResult içinde döndürür.
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


    if source_result.is_answerable != target_result.is_answerable:
        raise ValueError(
            f"Answerability mismatch inside pair '{source_result.pair_id}'."
        )

    source_reliable = is_reliable_outcome(
        source_result.outcome
    )

    target_reliable = is_reliable_outcome(
        target_result.outcome
    )

    transition = classify_reliability_transition(
        source_reliable=source_reliable,
        target_reliable=target_reliable
    )

    return PairedReliabilityResult(
        pair_id=source_result.pair_id,
        task=source_result.task,
        source_language=source_language,
        target_language=target_language,
        source_item_id=source_result.item_id,
        target_item_id=target_result.item_id,
        source_outcome=source_result.outcome,
        target_outcome=target_result.outcome,
        source_reliable=source_reliable,
        target_reliable=target_reliable,
        transition=transition,
        metadata=dict(source_result.metadata)
    )


def evaluate_paired_reliability(
    results: list[AbstentionResult],
    source_language: str = 'en',
    target_language: str = 'az'
) -> list[PairedReliabilityResult]:
    """Tüm AbstentionResult kayıtlarını pair seviyesinde EN-AZ olarak değerlendirir.

    Önce sonuçları `pair_id` değerine göre gruplar, sonra her pair için
    source ve target reliability karşılaştırması oluşturur.

    Örnek:
    reasoning_001_en → reliable=True
    reasoning_001_az → reliable=False

    reasoning_002_en → reliable=True
    reasoning_002_az → reliable=True

    Sonuç:
    [
        PairedReliabilityResult(
            pair_id="reasoning_001",
            transition="source_only_reliable",
            ...
        ),
        PairedReliabilityResult(
            pair_id="reasoning_002",
            transition="both_reliable",
            ...
        )
    ]

    Pair sırası input içinde ilk görüldüğü sıraya göre korunur.
    """

    grouped = group_reliability_by_pair(results)

    return [
        create_paired_reliability_result(
            pair_results=pair_results,
            source_language=source_language,
            target_language=target_language
        )
        for pair_results in grouped.values()
    ]


def summarize_paired_reliability(
    results: list[PairedReliabilityResult]
) -> dict[str, Any]:
    """Pair-level reliability transition dağılımını özetler.

    Döndürülen alanlar:
        total_pairs:
            Toplam EN-AZ pair sayısı.

        both_reliable:
            Her iki dilde reliable pair sayısı.

        source_only_reliable:
            Yalnızca source dilde reliable pair sayısı.

        target_only_reliable:
            Yalnızca target dilde reliable pair sayısı.

        both_unreliable:
            Her iki dilde unreliable pair sayısı.

        degradation_rate:
            source_only_reliable / total_pairs

        recovery_rate:
            target_only_reliable / total_pairs

        consistency_rate:
            Her iki dilin aynı reliability durumunu gösterdiği pair oranı.
    """

    if not results:
        raise ValueError(
            "Cannot summarize empty paired reliability results."
        )

    transition_counts = {
        "both_reliable": 0,
        "source_only_reliable": 0,
        "target_only_reliable": 0,
        "both_unreliable": 0
    }

    for result in results:
        if result.transition not in transition_counts:
            raise ValueError(
                f"Unknown reliability transition: '{result.transition}'"
            )

        transition_counts[result.transition] += 1

    total_pairs = len(results)

    consistent_pairs = (
        transition_counts['both_reliable']
        + transition_counts ['both_unreliable']
    )


    return {
        "total_pairs": total_pairs,
        **transition_counts,
        # degradation → AZ'de bozuldu
        "degradation_rate": (
            transition_counts["source_only_reliable"]
            / total_pairs
        ),
        # recovery → AZ'de düzeldi
        "recovery_rate": (
            transition_counts["target_only_reliable"]
            / total_pairs
        ),
        # consistency → iki dil aynı davrandı
        "consistency_rate": (
            consistent_pairs
            / total_pairs
        )
    }


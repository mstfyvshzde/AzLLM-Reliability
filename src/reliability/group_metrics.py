"""Abstention reliability sonuçlarını language ve task bazında özetler.

Bu modül item-level AbstentionResult kayıtlarını araştırma açısından
anlamlı gruplara ayırır.

Amaç özellikle English ve Azerbaijani arasında şu reliability davranışlarını
karşılaştırabilmektir:

- abstention accuracy
- over-answering rate
- under-answering rate
- empty-response rate

Bu çıktılar capability metricleriyle birlikte değerlendirildiğinde modelin
Azerbaycancaya geçerken yalnızca doğruluk kaybı mı yaşadığı, yoksa reliability
davranışının da bozulup bozulmadığı analiz edilebilir.
"""


from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from src.reliability.abstention_metrics import (
    CORRECT_ABSTENTION,
    CORRECT_ANSWER,
    EMPTY_RESPONSE,
    OVER_ANSWERING,
    UNDER_ANSWERING,
    AbstentionResult
)


# GroupKeyFunction → Bir AbstentionResult alıp hangi gruba ait olduğunu belirleyen string bir değer döndüren fonksiyon tipidir.
GroupKeyFunction = Callable[
    [AbstentionResult],
    str
]



def summarize_reliability_group(
    results: list[AbstentionResult]
) -> dict[str, Any]:
    """Tek bir abstention reliability grubunun metric özetini oluşturur.

    Döndürülen alanlar:
        total:
            Toplam item sayısı.

        answerable_total:
            Cevaplanabilir item sayısı.

        unanswerable_total:
            Cevaplanamaz item sayısı.

        correct_answer:
            Answerable item üzerinde cevap verilen kayıt sayısı.

        correct_abstention:
            Unanswerable item üzerinde doğru abstention sayısı.

        under_answering:
            Answerable item üzerinde gereksiz abstention sayısı.

        over_answering:
            Unanswerable item üzerinde cevap verilen kayıt sayısı.

        empty_response:
            Boş model çıktısı sayısı.

        abstention_accuracy:
            Doğru answer/abstention kararlarının tüm item'lara oranı.

        over_answering_rate:
            Unanswerable item'larda cevap verme oranı.

        under_answering_rate:
            Answerable item'larda gereksiz abstain etme oranı.

        empty_response_rate:
            Boş response oranı.
    """

    if not results: 
        raise ValueError(
            "Cannot summarize an empty reliability group."
        )

    counts = {
        CORRECT_ANSWER: 0,
        CORRECT_ABSTENTION: 0,
        UNDER_ANSWERING: 0,
        OVER_ANSWERING: 0,
        EMPTY_RESPONSE: 0
    }

    for result in results:
        if result.outcome not in counts:
            raise ValueError(
                f"Unknown abstention outcome: '{result.outcome}'"
            )

        counts[result.outcome] += 1

    total = len(results)

    answerable_total = sum(
        1
        for result in results
        if result.is_answerable
    )

    unanswerable_total = (total - answerable_total)

    correct_decisions = (
        counts[CORRECT_ANSWER]
        + counts[CORRECT_ABSTENTION]
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
        "answerable_total": answerable_total,
        "unanswerable_total": unanswerable_total,
        "correct_answer": counts[CORRECT_ANSWER],
        "correct_abstention": counts[CORRECT_ABSTENTION],
        "under_answering": counts[UNDER_ANSWERING],
        "over_answering": counts[OVER_ANSWERING],
        "empty_response": counts[EMPTY_RESPONSE],
        "abstention_accuracy": (
            correct_decisions
            / total
        ),
        "over_answering_rate": over_answering_rate,
        "under_answering_rate": under_answering_rate,
        "empty_response_rate": (
            counts[EMPTY_RESPONSE]
            / total
        )
    }



def group_reliability_results(
    results: list[AbstentionResult],
    key_function: GroupKeyFunction
) -> dict[str, list[AbstentionResult]]:
    """AbstentionResult kayıtlarını verilen key fonksiyonuna göre gruplar.

    `key_function`, her kaydın hangi gruba ait olduğunu belirler.

    Örnek:
    key_function = lambda result: result.language

    Input:
        result_1.language = "en"
        result_2.language = "az"
        result_3.language = "en"

    Sonuç:
        {
            "en": [result_1, result_3],
            "az": [result_2]
        }

    Böylece reliability sonuçları language, task veya başka bir özelliğe
    göre gruplanabilir.
    """

    grouped: dict[
        str,
        list[AbstentionResult]
    ] = defaultdict(list)

    for result in results:
        group_key = key_function(result)

        grouped[group_key].append(result)

    return dict(grouped)



def summarize_reliability_by_language(
    results: list[AbstentionResult]
) -> dict[str, dict[str, Any]]:
    """Reliability sonuçlarını dile göre gruplar ve her dil için özet metrikler üretir.

    Örnek:
    Input:
        en → correct_answer
        en → over_answering
        az → correct_abstention
        az → under_answering

    Önce gruplar:
    {
        "en": [result_1, result_2],
        "az": [result_3, result_4]
    }

    Sonra her grup için `summarize_reliability_group()` çalıştırılır.

    Sonuç:
    {
        "en": {
            "abstention_accuracy": ...,
            "over_answering_rate": ...,
            "under_answering_rate": ...
        },
        "az": {
            "abstention_accuracy": ...,
            "over_answering_rate": ...,
            "under_answering_rate": ...
        }
    }

    Böylece EN ve AZ reliability davranışı doğrudan karşılaştırılabilir.
    """

    grouped = group_reliability_results(
        results,
        key_function=lambda result: result.language
    )

    return {
        language: summarize_reliability_group(
            language_results
        )
        for language, language_results in grouped.items()
    }


def summarize_reliability_by_task(
    results: list[AbstentionResult]
) -> dict[str, dict[str, Any]]:
    """Reliability sonuçlarını benchmark task türüne göre gruplar ve özetler.

    Örnek:
    Input:
        reasoning → correct_answer
        reasoning → over_answering
        factual_qa → correct_abstention
        factual_qa → under_answering

    Önce gruplar:
    {
        "reasoning": [result_1, result_2],
        "factual_qa": [result_3, result_4]
    }

    Sonra her task grubu için `summarize_reliability_group()` çalıştırılır.

    Sonuç:
    {
        "reasoning": {
            "abstention_accuracy": ...,
            "over_answering_rate": ...,
            "under_answering_rate": ...
        },
        "factual_qa": {
            "abstention_accuracy": ...,
            "over_answering_rate": ...,
            "under_answering_rate": ...
        }
    }

    Böylece modelin reliability davranışının hangi task türlerinde daha iyi
    veya daha kötü olduğu görülebilir.
    """

    grouped = group_reliability_results(
        results,
        key_function=lambda result: result.task
    )

    return {
        task: summarize_reliability_group(
            task_results
        )
        for task, task_results in grouped.items()
    }


def summarize_reliability_by_category(
    results: list[AbstentionResult]
) -> dict[str, dict[str, Any]]:
    """Reliability sonuçlarını metadata içindeki category değerine göre gruplar ve özetler.

    Örnek:
    logical_reasoning → correct_answer, under_answering
    arithmetic        → correct_abstention, over_answering

    Önce gruplar:
    {
        "logical_reasoning": [result_1, result_2],
        "arithmetic": [result_3, result_4]
    }

    Sonra her category grubu için `summarize_reliability_group()` çalıştırılır.

    Category alanı olmayan kayıtlar:
    "__missing__"

    grubuna alınır.

    Böylece reliability problemlerinin hangi category'lerde daha fazla
    ortaya çıktığı görülebilir.
    """

    grouped = group_reliability_results(
        results,
        key_function=lambda result: str(
            result.metadata.get(
                'category',
                '__missing__'
            )
        )
    )

    return {
        category: summarize_reliability_group(category_results)
        for category, category_results in grouped.items()
    }


def summarize_reliability_by_difficulty(
    results: list[AbstentionResult]
) -> dict[str, dict[str, Any]]:
    """Reliability sonuçlarını difficulty seviyesine göre gruplar ve özetler.

    Örnek:
    easy   → correct_answer, correct_abstention
    medium → under_answering
    hard   → over_answering, empty_response

    Önce gruplar:
    {
        "easy": [result_1, result_2],
        "medium": [result_3],
        "hard": [result_4, result_5]
    }

    Sonra her difficulty grubu için `summarize_reliability_group()` çalıştırılır.

    Difficulty alanı olmayan kayıtlar:
    "__missing__"

    grubuna alınır.

    Böylece modelin reliability davranışının kolay, orta ve zor sorularda
    nasıl değiştiği görülebilir.
    """

    grouped = group_reliability_results(
        results,
        key_function=lambda result: str(
            result.metadata.get(
                'difficulty',
                '__missing__'
            )
        )
    )

    return {
        difficulty: summarize_reliability_group(
            difficulty_results
        )
        for difficulty, difficulty_results in grouped.items()
    }


def calculate_reliability_language_gap(
    results: list[AbstentionResult],
    source_language: str = 'en',
    target_language: str = 'az'
) -> dict[str, float]:
    """English ve Azerbaijani reliability metrikleri arasındaki farkı hesaplar.

    Önce reliability sonuçlarını language bazında özetler, sonra source language
    ve target language metriklerini karşılaştırır.

    Varsayılan:
    source_language = "en"
    target_language = "az"

    Örnek:
    EN abstention_accuracy = 0.85
    AZ abstention_accuracy = 0.70

    abstention_accuracy_gap:
        0.85 - 0.70 = 0.15

    Aynı şekilde:
    - over_answering_rate
    - under_answering_rate
    - empty_response_rate

    için de source - target farkı hesaplanır.

    Not:
    Abstention accuracy için yüksek değer genelde daha iyidir.
    Over-answering ve under-answering rate için düşük değer genelde daha iyidir.

    Source veya target language sonucu yoksa ValueError oluşturur.
    """

    summary = summarize_reliability_by_language(results)

    if source_language not in summary:
        raise ValueError(
            f"Missing source language results: '{source_language}'"
        )

    if target_language not in summary:
        raise ValueError(
            f"Missing target language results: '{target_language}'"
        )

    source = summary[source_language]
    target = summary[target_language]
    

    return {
        "source_abstention_accuracy": source[
            "abstention_accuracy"
        ],
        "target_abstention_accuracy": target[
            "abstention_accuracy"
        ],
        "abstention_accuracy_gap": (
            source["abstention_accuracy"]
            - target["abstention_accuracy"]
        ),
        "source_over_answering_rate": source[
            "over_answering_rate"
        ],
        "target_over_answering_rate": target[
            "over_answering_rate"
        ],
        "over_answering_rate_gap": (
            source["over_answering_rate"]
            - target["over_answering_rate"]
        ),
        "source_under_answering_rate": source[
            "under_answering_rate"
        ],
        "target_under_answering_rate": target[
            "under_answering_rate"
        ],
        "under_answering_rate_gap": (
            source["under_answering_rate"]
            - target["under_answering_rate"]
        ),
        "source_empty_response_rate": source[
            "empty_response_rate"
        ],
        "target_empty_response_rate": target[
            "empty_response_rate"
        ],
        "empty_response_rate_gap": (
            source["empty_response_rate"]
            - target["empty_response_rate"]
        )
    }
"""Exact-match sonuçlarını araştırma açısından anlamlı gruplara göre özetler.

Bu modül item-level ExactMatchResult kayıtlarını language, task, category ve
difficulty gibi benchmark özelliklerine göre gruplar.

Amaç yalnızca overall accuracy vermek değil, capability gap'in hangi dilde,
hangi task ailesinde ve hangi difficulty seviyesinde ortaya çıktığını
gösterebilecek aggregate metrikler üretmektir.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from src.evaluation.exact_match import ExactMatchResult

# GroupKeyFunction → Bir ExactMatchResult alıp hangi gruba ait olduğunu belirleyen string bir anahtar döndüren fonksiyon tipidir.
# Örnek: lambda result: result.language
# input: ExactMatchResult(language="en", ...)
# output: "en"
# Böylece sonuçlar "en", "az" gibi gruplara ayrılabilir.
GroupKeyFunction = Callable[
    [ExactMatchResult],
    str
]



def summarize_group(
    results: list[ExactMatchResult]
) -> dict[str, Any]:
    """Tek bir result grubunun temel exact-match özetini oluşturur.

    Döndürülen alanlar:
        total:
            Gruptaki toplam item sayısı.

        correct:
            Exact-match doğru item sayısı.

        incorrect:
            Exact-match yanlış item sayısı.

        accuracy:
            Correct / total oranı.

    Boş grup için ValueError oluşturur.
    """

    if not results:
        raise ValueError(
            "Cannot summarize an empty result group."
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



def group_results(
    results: list[ExactMatchResult],
    key_function: GroupKeyFunction
) -> dict[str, list[ExactMatchResult]]:
    """Exact-match sonuçlarını verilen key fonksiyonuna göre gruplar.

    key_function her ExactMatchResult için string bir group key üretmelidir.

    Örnek:
        lambda result: result.language

    Sonuç:
        {
            "en": [...],
            "az": [...]
        }
    """

    grouped: dict[
        str,
        list[ExactMatchResult]
    ] = defaultdict(list)

    for result in results:
        group_key = key_function(result)
        grouped[group_key].append(result)

    return dict(grouped)



def summarize_by_language(
    results: list[ExactMatchResult]
) -> dict[str, dict[str, Any]]:
    """Exact-match sonuçlarını language bazında özetler.

    Beklenen tipik gruplar:
        en
        az

    Bu çıktı daha sonra doğrudan EN-AZ capability gap hesaplamasında
    kullanılabilir.
    """

    grouped = group_results(
        results,
        key_function=lambda result: result.language
    )

    return {
        language: summarize_group(language_results)
        for language, language_results in grouped.items()
    }


def summarize_by_task(
    results: list[ExactMatchResult]
) -> dict[str, dict[str, Any]]:
    """Exact-match sonuçlarını task türüne göre gruplar ve her task için özet metrik üretir.

    Örnek:
    reasoning → [1, 1, 0, 1]
    factual_qa → [1, 0, 0]

    Sonuç:
    {
        "reasoning": {
            "total": 4,
            "correct": 3,
            "incorrect": 1,
            "accuracy": 0.75
        },
        "factual_qa": {
            "total": 3,
            "correct": 1,
            "incorrect": 2,
            "accuracy": 0.33
        }
    }

    Yani modelin hangi task ailesinde ne kadar başarılı olduğunu gösterir.
    """

    grouped = group_results(
        results,
        key_function=lambda result: result.task
    )

    return {
        task: summarize_group(task_results)
        for task, task_results in grouped.items()
    }



def summarize_by_category(
    results: list[ExactMatchResult]
) -> dict[str, dict[str, Any]]:
    """Exact-match sonuçlarını metadata içindeki category değerine göre gruplar ve özetler.

    Örnek:
    logical_reasoning → [1, 1, 0]
    arithmetic        → [1, 0]
    category yok      → [0]

    Sonuç:
    {
        "logical_reasoning": {
            "total": 3,
            "correct": 2,
            "incorrect": 1,
            "accuracy": 0.67
        },
        "arithmetic": {
            "total": 2,
            "correct": 1,
            "incorrect": 1,
            "accuracy": 0.5
        },
        "__missing__": {
            ...
        }
    }

    Category alanı olmayan kayıtlar `__missing__` grubunda tutulur.
    """

    grouped = group_results(
        results,
        key_function=lambda result: str(
            result.metadata.get(
                'category',
                '__missing__'
            )
        )
    )

    return {
        category: summarize_group(category_results)
        for category, category_results in grouped.items()
    }


def summarize_by_difficulty(
    results: list[ExactMatchResult]
) -> dict[str, dict[str, Any]]:
    """Exact-match sonuçlarını difficulty seviyesine göre gruplar ve özetler.

    Örnek:
    easy   → [1, 1, 1]
    medium → [1, 0, 1]
    hard   → [0, 0, 1]

    Sonuç:
    {
        "easy": {
            "total": 3,
            "correct": 3,
            "incorrect": 0,
            "accuracy": 1.0
        },
        "medium": {
            "total": 3,
            "correct": 2,
            "incorrect": 1,
            "accuracy": 0.67
        },
        "hard": {
            "total": 3,
            "correct": 1,
            "incorrect": 2,
            "accuracy": 0.33
        }
    }

    Difficulty alanı olmayan kayıtlar `__missing__` grubunda tutulur.
    """

    grouped = group_results(
        results,
        key_function=lambda result: str(
            result.metadata.get(
                'difficulty',
                '__missing__'
            )
        )
    )

    return {
        difficulty: summarize_group(difficulty_results)
        for difficulty, difficulty_results in grouped.items()
    }


def summarize_language_task_matrix(
    results: list[ExactMatchResult]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Language ve task kombinasyonlarına göre nested metric özeti üretir.

    Örnek çıktı:

        {
            "en": {
                "reasoning": {
                    "total": 10,
                    "correct": 8,
                    "incorrect": 2,
                    "accuracy": 0.8
                }
            },
            "az": {
                "reasoning": {
                    "total": 10,
                    "correct": 6,
                    "incorrect": 4,
                    "accuracy": 0.6
                }
            }
        }

    Bu yapı aynı task için EN ve AZ performansını doğrudan karşılaştırmayı
    kolaylaştırır.
    """

    matrix: dict[
        str,
        dict[
            str,
            list[ExactMatchResult]
        ]
    ] = defaultdict(lambda: defaultdict(list))


    for result in results:
        matrix[
            result.language
        ][
            result.task
        ].append(
            result
        )

    return {
        language: {
            task: summarize_group(
                task_results
            )
            for task, task_results in task_groups.items()
        }
        for language, task_groups in matrix.items()
    }


def calculate_language_gap(
    results: list[ExactMatchResult],
    source_language: str = 'en',
    target_language: str = 'az'
) -> dict[str, float]:
    """Source ve target language accuracy arasındaki capability gap'i hesaplar.

    `source_language`, karşılaştırmadaki kaynak dili temsil eder.
    Bu projede varsayılan olarak İngilizce (`en`) kullanılır.

    `target_language`, karşılaştırılan hedef dili temsil eder.
    Bu projede varsayılan olarak Azerbaycanca (`az`) kullanılır.

    Örnek:
    source_language = "en"
    target_language = "az"

    English accuracy = 0.80
    Azerbaijani accuracy = 0.60

    absolute_gap = 0.80 - 0.60 = 0.20

    Pozitif gap:
    Source language daha yüksek performans göstermiştir.

    Negatif gap:
    Target language daha yüksek performans göstermiştir.

    Her iki language grubu mevcut değilse ValueError oluşturur.
    """

    language_summary = summarize_by_language(results)

    if source_language not in language_summary:
        raise ValueError(
            f"Missing source language results: '{target_language}'"
        )

    source_accuracy = language_summary[
        source_language
    ] [
        'accuracy'
    ]

    if target_language not in language_summary:
        raise ValueError(
            f"Missing target language results: '{target_language}'"
        )

    target_accuracy = language_summary[
        target_language
    ] [
        'accuracy'
    ]

    return {
        "source_accuracy": source_accuracy,
        "target_accuracy": target_accuracy,
        "absolute_gap": (
            source_accuracy
            - target_accuracy
        )
    }
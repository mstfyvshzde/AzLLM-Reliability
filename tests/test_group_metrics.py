"""Grouped capability metric yardımcılarını test eder."""

import pytest

from src.evaluation.exact_match import ExactMatchResult
from src.evaluation.group_metrics import (
    calculate_language_gap,
    group_results,
    summarize_by_category,
    summarize_by_difficulty,
    summarize_by_language,
    summarize_by_task,
    summarize_group,
    summarize_language_task_matrix,
)


def make_result(
    item_id: str,
    pair_id: str,
    language: str,
    task: str = "reasoning",
    exact_match: int = 1,
    category: str = "arithmetic_reasoning",
    difficulty: str = "easy",
) -> ExactMatchResult:
    """Grouped metric testleri için örnek ExactMatchResult oluşturur."""

    prediction = "4" if exact_match == 1 else "5"
    reference_answer = "4"

    return ExactMatchResult(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        prediction=prediction,
        reference_answer=reference_answer,
        normalized_prediction=prediction,
        normalized_reference=reference_answer,
        exact_match=exact_match,
        metadata={
            "category": category,
            "difficulty": difficulty,
            "review_status": "approved",
        },
    )


def test_summarize_group() -> None:
    """Tek result grubunun aggregate özetinin doğru hesaplandığını test eder."""

    results = [
        make_result(
            item_id="item_1",
            pair_id="pair_1",
            language="en",
            exact_match=1,
        ),
        make_result(
            item_id="item_2",
            pair_id="pair_2",
            language="en",
            exact_match=0,
        ),
        make_result(
            item_id="item_3",
            pair_id="pair_3",
            language="en",
            exact_match=1,
        ),
    ]

    summary = summarize_group(
        results
    )

    assert summary["total"] == 3
    assert summary["correct"] == 2
    assert summary["incorrect"] == 1
    assert summary["accuracy"] == pytest.approx(
        2 / 3
    )


def test_summarize_group_rejects_empty_input() -> None:
    """Boş result grubunun summarize edilmesinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Cannot summarize an empty result group",
    ):
        summarize_group(
            []
        )


def test_group_results() -> None:
    """Result kayıtlarının verilen key fonksiyonuna göre gruplandığını test eder."""

    results = [
        make_result(
            item_id="item_en",
            pair_id="pair_1",
            language="en",
        ),
        make_result(
            item_id="item_az",
            pair_id="pair_1",
            language="az",
        ),
    ]

    grouped = group_results(
        results,
        key_function=lambda result: result.language,
    )

    assert set(grouped) == {
        "en",
        "az",
    }

    assert grouped["en"][0].item_id == "item_en"
    assert grouped["az"][0].item_id == "item_az"


def test_summarize_by_language() -> None:
    """EN ve AZ exact-match sonuçlarının ayrı accuracy değerleri ürettiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            exact_match=1,
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            exact_match=1,
        ),
        make_result(
            "item_3",
            "pair_1",
            "az",
            exact_match=1,
        ),
        make_result(
            "item_4",
            "pair_2",
            "az",
            exact_match=0,
        ),
    ]

    summary = summarize_by_language(
        results
    )

    assert summary["en"]["total"] == 2
    assert summary["en"]["accuracy"] == 1.0

    assert summary["az"]["total"] == 2
    assert summary["az"]["accuracy"] == 0.5


def test_summarize_by_task() -> None:
    """Result kayıtlarının task bazında özetlendiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            task="reasoning",
            exact_match=1,
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            task="factual_knowledge",
            exact_match=0,
        ),
    ]

    summary = summarize_by_task(
        results
    )

    assert summary["reasoning"]["accuracy"] == 1.0
    assert summary["factual_knowledge"]["accuracy"] == 0.0


def test_summarize_by_category() -> None:
    """Metadata category alanına göre metric özeti üretildiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            exact_match=1,
            category="arithmetic_reasoning",
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            exact_match=0,
            category="logical_reasoning",
        ),
    ]

    summary = summarize_by_category(
        results
    )

    assert summary[
        "arithmetic_reasoning"
    ][
        "accuracy"
    ] == 1.0

    assert summary[
        "logical_reasoning"
    ][
        "accuracy"
    ] == 0.0


def test_summarize_by_category_tracks_missing_metadata() -> None:
    """Eksik category metadata değerinin görünür __missing__ grubuna alındığını test eder."""

    result = make_result(
        "item_1",
        "pair_1",
        "en",
    )

    result.metadata.pop(
        "category"
    )

    summary = summarize_by_category(
        [result]
    )

    assert "__missing__" in summary
    assert summary["__missing__"]["total"] == 1


def test_summarize_by_difficulty() -> None:
    """Result kayıtlarının difficulty bazında özetlendiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            exact_match=1,
            difficulty="easy",
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            exact_match=0,
            difficulty="hard",
        ),
    ]

    summary = summarize_by_difficulty(
        results
    )

    assert summary["easy"]["accuracy"] == 1.0
    assert summary["hard"]["accuracy"] == 0.0


def test_summarize_by_difficulty_tracks_missing_metadata() -> None:
    """Eksik difficulty metadata değerinin __missing__ grubunda tutulduğunu test eder."""

    result = make_result(
        "item_1",
        "pair_1",
        "az",
    )

    result.metadata.pop(
        "difficulty"
    )

    summary = summarize_by_difficulty(
        [result]
    )

    assert summary["__missing__"]["total"] == 1


def test_summarize_language_task_matrix() -> None:
    """Language-task kombinasyonları için nested accuracy matrisi üretildiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            task="reasoning",
            exact_match=1,
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            task="reasoning",
            exact_match=0,
        ),
        make_result(
            "item_3",
            "pair_1",
            "az",
            task="reasoning",
            exact_match=1,
        ),
        make_result(
            "item_4",
            "pair_3",
            "az",
            task="factual_knowledge",
            exact_match=0,
        ),
    ]

    matrix = summarize_language_task_matrix(
        results
    )

    assert matrix[
        "en"
    ][
        "reasoning"
    ][
        "accuracy"
    ] == 0.5

    assert matrix[
        "az"
    ][
        "reasoning"
    ][
        "accuracy"
    ] == 1.0

    assert matrix[
        "az"
    ][
        "factual_knowledge"
    ][
        "accuracy"
    ] == 0.0


def test_calculate_language_gap() -> None:
    """Source-target accuracy farkının doğru hesaplandığını test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            exact_match=1,
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            exact_match=1,
        ),
        make_result(
            "item_3",
            "pair_1",
            "az",
            exact_match=1,
        ),
        make_result(
            "item_4",
            "pair_2",
            "az",
            exact_match=0,
        ),
    ]

    gap = calculate_language_gap(
        results
    )

    assert gap["source_accuracy"] == 1.0
    assert gap["target_accuracy"] == 0.5
    assert gap["absolute_gap"] == 0.5


def test_calculate_negative_language_gap() -> None:
    """Target language daha iyi olduğunda gap değerinin negatif olduğunu test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            exact_match=0,
        ),
        make_result(
            "item_2",
            "pair_2",
            "az",
            exact_match=1,
        ),
    ]

    gap = calculate_language_gap(
        results
    )

    assert gap["source_accuracy"] == 0.0
    assert gap["target_accuracy"] == 1.0
    assert gap["absolute_gap"] == -1.0


def test_calculate_language_gap_rejects_missing_source() -> None:
    """Source language sonuçları bulunmadığında gap hesabının reddedildiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "az",
            exact_match=1,
        )
    ]

    with pytest.raises(
        ValueError,
        match="Missing source language results",
    ):
        calculate_language_gap(
            results
        )


def test_calculate_language_gap_rejects_missing_target() -> None:
    """Target language sonuçları bulunmadığında gap hesabının reddedildiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            exact_match=1,
        )
    ]

    with pytest.raises(
        ValueError,
        match="Missing target language results",
    ):
        calculate_language_gap(
            results
        )
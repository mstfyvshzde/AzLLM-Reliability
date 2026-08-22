"""Reliability grouped metric yardımcılarını test eder."""

import pytest

from src.reliability.abstention_metrics import (
    CORRECT_ABSTENTION,
    CORRECT_ANSWER,
    EMPTY_RESPONSE,
    OVER_ANSWERING,
    UNDER_ANSWERING,
    AbstentionResult,
)
from src.reliability.group_metrics import (
    calculate_reliability_language_gap,
    group_reliability_results,
    summarize_reliability_by_category,
    summarize_reliability_by_difficulty,
    summarize_reliability_by_language,
    summarize_reliability_by_task,
    summarize_reliability_group,
)


def make_result(
    item_id: str,
    pair_id: str,
    language: str,
    is_answerable: bool,
    outcome: str,
    task: str = "reasoning",
    category: str = "arithmetic_reasoning",
    difficulty: str = "easy",
) -> AbstentionResult:
    """Reliability grouped metric testleri için örnek AbstentionResult oluşturur."""

    if outcome in {
        CORRECT_ANSWER,
        OVER_ANSWERING,
    }:
        response_status = "answered"
        prediction = "4"

    elif outcome in {
        CORRECT_ABSTENTION,
        UNDER_ANSWERING,
    }:
        response_status = "abstained"
        prediction = "I don't know."

    else:
        response_status = "empty"
        prediction = ""

    return AbstentionResult(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        prediction=prediction,
        is_answerable=is_answerable,
        response_status=response_status,
        outcome=outcome,
        metadata={
            "category": category,
            "difficulty": difficulty,
            "review_status": "approved",
            "is_answerable": is_answerable,
        },
    )


def test_summarize_reliability_group() -> None:
    """Tek reliability grubunun temel metriclerini doğru hesaplar."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            True,
            CORRECT_ANSWER,
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            True,
            UNDER_ANSWERING,
        ),
        make_result(
            "item_3",
            "pair_3",
            "en",
            False,
            CORRECT_ABSTENTION,
        ),
        make_result(
            "item_4",
            "pair_4",
            "en",
            False,
            OVER_ANSWERING,
        ),
    ]

    summary = summarize_reliability_group(
        results
    )

    assert summary["total"] == 4
    assert summary["answerable_total"] == 2
    assert summary["unanswerable_total"] == 2

    assert summary["correct_answer"] == 1
    assert summary["correct_abstention"] == 1
    assert summary["under_answering"] == 1
    assert summary["over_answering"] == 1
    assert summary["empty_response"] == 0

    assert summary["abstention_accuracy"] == pytest.approx(
        0.5
    )

    assert summary["over_answering_rate"] == pytest.approx(
        0.5
    )

    assert summary["under_answering_rate"] == pytest.approx(
        0.5
    )

    assert summary["empty_response_rate"] == 0.0


def test_summarize_reliability_group_with_empty_response() -> None:
    """Empty response oranının doğru hesaplandığını test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            True,
            EMPTY_RESPONSE,
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            True,
            CORRECT_ANSWER,
        ),
    ]

    summary = summarize_reliability_group(
        results
    )

    assert summary["empty_response"] == 1
    assert summary["empty_response_rate"] == pytest.approx(
        0.5
    )


def test_summarize_reliability_group_rejects_empty_input() -> None:
    """Boş reliability grubunun summarize edilmesini reddeder."""

    with pytest.raises(
        ValueError,
        match="Cannot summarize an empty reliability group",
    ):
        summarize_reliability_group(
            []
        )


def test_summarize_reliability_group_rejects_unknown_outcome() -> None:
    """Tanımsız reliability outcome değerinin reddedildiğini test eder."""

    result = make_result(
        "item_1",
        "pair_1",
        "en",
        True,
        CORRECT_ANSWER,
    )

    invalid_result = AbstentionResult(
        item_id=result.item_id,
        pair_id=result.pair_id,
        language=result.language,
        task=result.task,
        prediction=result.prediction,
        is_answerable=result.is_answerable,
        response_status=result.response_status,
        outcome="unknown",
        metadata=result.metadata,
    )

    with pytest.raises(
        ValueError,
        match="Unknown abstention outcome",
    ):
        summarize_reliability_group(
            [invalid_result]
        )


def test_group_reliability_results() -> None:
    """Reliability kayıtlarının verilen key fonksiyonuna göre gruplandığını test eder."""

    results = [
        make_result(
            "item_en",
            "pair_1",
            "en",
            True,
            CORRECT_ANSWER,
        ),
        make_result(
            "item_az",
            "pair_1",
            "az",
            True,
            CORRECT_ANSWER,
        ),
    ]

    grouped = group_reliability_results(
        results,
        key_function=lambda result: result.language,
    )

    assert set(grouped) == {
        "en",
        "az",
    }

    assert grouped["en"][0].item_id == "item_en"
    assert grouped["az"][0].item_id == "item_az"


def test_summarize_reliability_by_language() -> None:
    """EN ve AZ reliability metriclerinin ayrı hesaplandığını test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            True,
            CORRECT_ANSWER,
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            False,
            CORRECT_ABSTENTION,
        ),
        make_result(
            "item_3",
            "pair_1",
            "az",
            True,
            UNDER_ANSWERING,
        ),
        make_result(
            "item_4",
            "pair_2",
            "az",
            False,
            OVER_ANSWERING,
        ),
    ]

    summary = summarize_reliability_by_language(
        results
    )

    assert summary["en"]["abstention_accuracy"] == 1.0
    assert summary["az"]["abstention_accuracy"] == 0.0

    assert summary["en"]["over_answering_rate"] == 0.0
    assert summary["az"]["over_answering_rate"] == 1.0

    assert summary["en"]["under_answering_rate"] == 0.0
    assert summary["az"]["under_answering_rate"] == 1.0


def test_summarize_reliability_by_task() -> None:
    """Reliability sonuçlarının task bazında özetlendiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            True,
            CORRECT_ANSWER,
            task="reasoning",
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            False,
            OVER_ANSWERING,
            task="unanswerable",
        ),
    ]

    summary = summarize_reliability_by_task(
        results
    )

    assert summary["reasoning"]["abstention_accuracy"] == 1.0
    assert summary["unanswerable"]["abstention_accuracy"] == 0.0


def test_summarize_reliability_by_category() -> None:
    """Reliability sonuçlarının category bazında özetlendiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            True,
            CORRECT_ANSWER,
            category="arithmetic_reasoning",
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            False,
            OVER_ANSWERING,
            category="logical_reasoning",
        ),
    ]

    summary = summarize_reliability_by_category(
        results
    )

    assert summary[
        "arithmetic_reasoning"
    ][
        "abstention_accuracy"
    ] == 1.0

    assert summary[
        "logical_reasoning"
    ][
        "abstention_accuracy"
    ] == 0.0


def test_summarize_reliability_by_category_tracks_missing_metadata() -> None:
    """Eksik category metadata değerinin __missing__ grubunda tutulduğunu test eder."""

    result = make_result(
        "item_1",
        "pair_1",
        "en",
        True,
        CORRECT_ANSWER,
    )

    result.metadata.pop(
        "category"
    )

    summary = summarize_reliability_by_category(
        [result]
    )

    assert "__missing__" in summary
    assert summary["__missing__"]["total"] == 1


def test_summarize_reliability_by_difficulty() -> None:
    """Reliability sonuçlarının difficulty bazında özetlendiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            True,
            CORRECT_ANSWER,
            difficulty="easy",
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            False,
            OVER_ANSWERING,
            difficulty="hard",
        ),
    ]

    summary = summarize_reliability_by_difficulty(
        results
    )

    assert summary["easy"]["abstention_accuracy"] == 1.0
    assert summary["hard"]["abstention_accuracy"] == 0.0


def test_summarize_reliability_by_difficulty_tracks_missing_metadata() -> None:
    """Eksik difficulty metadata değerinin __missing__ grubunda tutulduğunu test eder."""

    result = make_result(
        "item_1",
        "pair_1",
        "az",
        True,
        CORRECT_ANSWER,
    )

    result.metadata.pop(
        "difficulty"
    )

    summary = summarize_reliability_by_difficulty(
        [result]
    )

    assert summary["__missing__"]["total"] == 1


def test_calculate_reliability_language_gap() -> None:
    """EN-AZ reliability metric farklarının doğru hesaplandığını test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            True,
            CORRECT_ANSWER,
        ),
        make_result(
            "item_2",
            "pair_2",
            "en",
            False,
            CORRECT_ABSTENTION,
        ),
        make_result(
            "item_3",
            "pair_1",
            "az",
            True,
            UNDER_ANSWERING,
        ),
        make_result(
            "item_4",
            "pair_2",
            "az",
            False,
            OVER_ANSWERING,
        ),
    ]

    gap = calculate_reliability_language_gap(
        results
    )

    assert gap["source_abstention_accuracy"] == 1.0
    assert gap["target_abstention_accuracy"] == 0.0
    assert gap["abstention_accuracy_gap"] == 1.0

    assert gap["source_over_answering_rate"] == 0.0
    assert gap["target_over_answering_rate"] == 1.0
    assert gap["over_answering_rate_gap"] == -1.0

    assert gap["source_under_answering_rate"] == 0.0
    assert gap["target_under_answering_rate"] == 1.0
    assert gap["under_answering_rate_gap"] == -1.0

    assert gap["source_empty_response_rate"] == 0.0
    assert gap["target_empty_response_rate"] == 0.0
    assert gap["empty_response_rate_gap"] == 0.0


def test_calculate_reliability_language_gap_rejects_missing_source() -> None:
    """Source language sonucu yoksa gap hesabının reddedildiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "az",
            True,
            CORRECT_ANSWER,
        )
    ]

    with pytest.raises(
        ValueError,
        match="Missing source language results",
    ):
        calculate_reliability_language_gap(
            results
        )


def test_calculate_reliability_language_gap_rejects_missing_target() -> None:
    """Target language sonucu yoksa gap hesabının reddedildiğini test eder."""

    results = [
        make_result(
            "item_1",
            "pair_1",
            "en",
            True,
            CORRECT_ANSWER,
        )
    ]

    with pytest.raises(
        ValueError,
        match="Missing target language results",
    ):
        calculate_reliability_language_gap(
            results
        )
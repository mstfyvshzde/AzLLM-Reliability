"""Benchmark EN-AZ pairing kurallarını test eder."""

import pytest

from src.data.benchmark_record import BenchmarkRecord
from src.data.pairing import (
    group_by_pair,
    validate_complete_pairs,
    validate_pair_task_consistency
)


def make_record(
    item_id: str,
    pair_id: str,
    language: str,
    task: str = "reasoning"
) -> BenchmarkRecord:
    """Pairing testleri için örnek benchmark kaydı oluşturur."""
    return BenchmarkRecord(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        question="Example question",
        reference_answer="Example answer"
    )


def test_group_by_pair() -> None:
    """Kayıtların pair_id değerine göre gruplandığını test eder."""
    records = [
        make_record("reasoning_001_en", "reasoning_001", "en"),
        make_record("reasoning_001_az", "reasoning_001", "az"),
        make_record("reasoning_002_en", "reasoning_002", "en"),
        make_record("reasoning_002_az", "reasoning_002", "az")
    ]

    pairs = group_by_pair(records)

    assert len(pairs) == 2
    assert len(pairs["reasoning_001"]) == 2
    assert len(pairs["reasoning_002"]) == 2


def test_complete_pair_is_valid() -> None:
    """Tam EN-AZ pair yapısının kabul edildiğini test eder."""
    records = [
        make_record("reasoning_001_en", "reasoning_001", "en"),
        make_record("reasoning_001_az", "reasoning_001", "az")
    ]

    validate_complete_pairs(records, {"en", "az"})


def test_incomplete_pair_is_rejected() -> None:
    """Dil karşılığı eksik olan pair'in reddedildiğini test eder."""
    records = [
        make_record("reasoning_001_en", "reasoning_001", "en")
    ]

    with pytest.raises(ValueError, match="Incomplete pair"):
        validate_complete_pairs(records, {"en", "az"})


def test_consistent_pair_tasks_are_valid() -> None:
    """EN ve AZ kayıtlarının aynı task değerine sahip olmasını test eder."""
    records = [
        make_record(
            "reasoning_001_en",
            "reasoning_001",
            "en",
            task="reasoning"
        ),
        make_record(
            "reasoning_001_az",
            "reasoning_001",
            "az",
            task="reasoning"
        )
    ]

    validate_pair_task_consistency(records)


def test_inconsistent_pair_tasks_are_rejected() -> None:
    """Aynı pair içinde farklı task değerlerinin reddedildiğini test eder."""
    records = [
        make_record(
            "reasoning_001_en",
            "reasoning_001",
            "en",
            task="reasoning"
        ),
        make_record(
            "reasoning_001_az",
            "reasoning_001",
            "az",
            task="factual_knowledge"
        )
    ]

    with pytest.raises(ValueError, match="Inconsistent task values"):
        validate_pair_task_consistency(records)
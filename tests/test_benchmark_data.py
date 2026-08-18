"""Benchmark veri yapısı, doğrulama ve pairing kurallarını test eder."""

import pytest

from src.data.benchmark_record import BenchmarkRecord
from src.data.pairing import validate_complete_pairs
from src.data.validate_benchmark import (
    validate_record,
    validate_unique_item_ids,
    validate_unique_pair_languages,
)


def make_record(
    item_id: str,
    pair_id: str,
    language: str
) -> BenchmarkRecord:
    """Testlerde kullanılacak örnek benchmark kaydı oluşturur."""

    return BenchmarkRecord(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task="reasoning",
        question="Example question",
        reference_answer="Example answer"
    )


def test_valid_record() -> None:
    """Geçerli bir benchmark kaydının doğrulamadan geçtiğini test eder."""

    record = make_record(
        item_id="reasoning_001_en",
        pair_id="reasoning_001",
        language="en"
    )

    validate_record(record, {"en", "az"})


def test_empty_question_is_rejected() -> None:
    """Boş question alanının reddedildiğini test eder."""

    record = BenchmarkRecord(
        item_id="reasoning_001_en",
        pair_id="reasoning_001",
        language="en",
        task="reasoning",
        question="",
        reference_answer="5"
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        validate_record(record, {"en", "az"})


def test_unsupported_language_is_rejected() -> None:
    """Desteklenmeyen dil kodunun reddedildiğini test eder."""

    record = make_record(
        item_id="reasoning_001_tr",
        pair_id="reasoning_001",
        language="tr"
    )

    with pytest.raises(ValueError, match="Unsupported language"):
        validate_record(record, {"en", "az"})


def test_duplicate_item_ids_are_rejected() -> None:
    """Tekrar eden item_id değerlerinin reddedildiğini test eder."""
    
    records = [
        make_record("reasoning_001_en", "reasoning_001", "en"),
        make_record("reasoning_001_en", "reasoning_002", "en")
    ]

    with pytest.raises(ValueError, match="Duplicate item_id"):
        validate_unique_item_ids(records)


def test_duplicate_pair_language_is_rejected() -> None:
    """Aynı pair_id-language kombinasyonunun tekrarını test eder."""

    records = [
        make_record("reasoning_001_en", "reasoning_001", "en"),
        make_record("reasoning_001_en_v2", "reasoning_001", "en")
    ]

    with pytest.raises(ValueError, match="Duplicate pair-language"):
        validate_unique_pair_languages(records)


def test_complete_bilingual_pair() -> None:
    """Tam EN-AZ pair yapısının doğrulamadan geçtiğini test eder."""

    records = [
        make_record("reasoning_001_en", "reasoning_001", "en"),
        make_record("reasoning_001_az", "reasoning_001", "az")
    ]

    validate_complete_pairs(records, {"en", "az"})


def test_incomplete_pair_is_rejected() -> None:
    """Eksik dil karşılığı bulunan pair'in reddedildiğini test eder."""
    
    records = [
        make_record("reasoning_001_en", "reasoning_001", "en")
    ]

    with pytest.raises(ValueError, match="Incomplete pair"):
        validate_complete_pairs(records, {"en", "az"})
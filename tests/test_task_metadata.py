"""Task-specific metadata doğrulama kurallarını test eder."""

import pytest

from src.data.benchmark_record import BenchmarkRecord
from src.data.validate_task_metadata import (
    validate_category,
    validate_difficulty,
    validate_required_metadata,
    validate_task_metadata,
)


def make_spec() -> dict:
    """Testlerde kullanılacak minimal reasoning task specification oluşturur."""
    return {
        "task": {
            "name": "reasoning",
        },
        "categories": {
            "logical_reasoning": {
                "enabled": True,
            },
            "arithmetic_reasoning": {
                "enabled": True,
            },
            "disabled_category": {
                "enabled": False,
            },
        },
        "difficulty": {
            "levels": [
                "easy",
                "medium",
                "hard",
            ],
        },
        "metadata": {
            "required_fields": [
                "category",
                "difficulty",
                "review_status",
            ],
        },
    }


def make_record(
    *,
    task: str = "reasoning",
    category: str = "logical_reasoning",
    difficulty: str = "easy",
    review_status: str = "pending",
) -> BenchmarkRecord:
    """Task metadata testleri için örnek benchmark kaydı oluşturur."""
    return BenchmarkRecord(
        item_id="reasoning_0001_en",
        pair_id="reasoning_0001",
        language="en",
        task=task,
        question="Example question",
        reference_answer="Example answer",
        metadata={
            "category": category,
            "difficulty": difficulty,
            "review_status": review_status,
        },
    )


def test_valid_task_metadata() -> None:
    """Geçerli task metadata yapısının doğrulamadan geçtiğini test eder."""
    validate_task_metadata(
        make_record(),
        make_spec(),
    )


def test_missing_required_metadata_is_rejected() -> None:
    """Zorunlu metadata alanı eksikse kaydın reddedildiğini test eder."""
    record = make_record()

    record.metadata.pop("difficulty")

    with pytest.raises(
        ValueError,
        match="Missing required metadata",
    ):
        validate_required_metadata(
            record,
            make_spec(),
        )


def test_unknown_category_is_rejected() -> None:
    """Tanımlanmamış category değerinin reddedildiğini test eder."""
    record = make_record(
        category="unknown_category",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported category",
    ):
        validate_category(
            record,
            make_spec(),
        )


def test_disabled_category_is_rejected() -> None:
    """Disabled category değerinin reddedildiğini test eder."""
    record = make_record(
        category="disabled_category",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported category",
    ):
        validate_category(
            record,
            make_spec(),
        )


def test_invalid_difficulty_is_rejected() -> None:
    """İzin verilmeyen difficulty değerinin reddedildiğini test eder."""
    record = make_record(
        difficulty="extreme",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported difficulty",
    ):
        validate_difficulty(
            record,
            make_spec(),
        )


def test_task_spec_mismatch_is_rejected() -> None:
    """Record task ile specification task farklıysa kaydın reddedildiğini test eder."""
    record = make_record(
        task="factual_knowledge",
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        validate_task_metadata(
            record,
            make_spec(),
        )
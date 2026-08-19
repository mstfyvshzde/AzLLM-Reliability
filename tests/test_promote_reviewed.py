"""Reviewed benchmark kayıtlarının final raw benchmark'a aktarılmasını test eder."""

import json
from pathlib import Path

import pytest

from src.data.promote_reviewed import (
    collect_reviewed_records,
    save_final_records,
    validate_promoted_records,
)
from src.data.benchmark_record import BenchmarkRecord


def make_task_specifications() -> dict[str, dict]:
    """Promotion testlerinde kullanılacak minimal task specification oluşturur."""
    return {
        "reasoning": {
            "task": {
                "name": "reasoning",
            },
            "categories": {
                "arithmetic_reasoning": {
                    "enabled": True,
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
    }


def make_record(
    item_id: str,
    pair_id: str,
    language: str,
    status: str = "approved",
) -> BenchmarkRecord:
    """Promotion testleri için örnek benchmark kaydı oluşturur."""
    return BenchmarkRecord(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task="reasoning",
        question="Example question",
        reference_answer="Example answer",
        metadata={
            "category": "arithmetic_reasoning",
            "difficulty": "easy",
            "review_status": status,
        },
    )


def test_collect_reviewed_records(tmp_path: Path) -> None:
    """Reviewed klasöründeki JSONL kayıtlarının toplandığını test eder."""
    input_path = tmp_path / "reasoning.jsonl"

    input_path.write_text(
        json.dumps(
            make_record(
                "reasoning_0001_en",
                "reasoning_0001",
                "en",
            ).to_dict()
        )
        + "\n"
        + json.dumps(
            make_record(
                "reasoning_0001_az",
                "reasoning_0001",
                "az",
            ).to_dict()
        )
        + "\n",
        encoding="utf-8",
    )

    records = collect_reviewed_records(tmp_path)

    assert len(records) == 2
    assert records[0].pair_id == "reasoning_0001"
    assert records[1].pair_id == "reasoning_0001"


def test_empty_reviewed_directory_is_rejected(
    tmp_path: Path,
) -> None:
    """JSONL bulunmayan reviewed klasörünün reddedildiğini test eder."""
    with pytest.raises(ValueError, match="No reviewed JSONL files"):
        collect_reviewed_records(tmp_path)


def test_valid_promoted_records() -> None:
    """Geçerli approved EN-AZ pair'in promotion doğrulamasından geçtiğini test eder."""
    records = [
        make_record(
            "reasoning_0001_en",
            "reasoning_0001",
            "en",
        ),
        make_record(
            "reasoning_0001_az",
            "reasoning_0001",
            "az",
        ),
    ]

    validate_promoted_records(
        records,
        make_task_specifications(),
    )


def test_empty_promoted_records_are_rejected() -> None:
    """Boş approved record listesinin reddedildiğini test eder."""
    with pytest.raises(ValueError, match="No approved records"):
        validate_promoted_records(
            [],
            make_task_specifications(),
        )


def test_duplicate_item_ids_are_rejected() -> None:
    """Promotion sırasında duplicate item_id değerlerinin reddedildiğini test eder."""
    records = [
        make_record(
            "reasoning_0001_en",
            "reasoning_0001",
            "en",
        ),
        make_record(
            "reasoning_0001_en",
            "reasoning_0001",
            "az",
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate item_id"):
        validate_promoted_records(
            records,
            make_task_specifications(),
        )


def test_save_final_records(tmp_path: Path) -> None:
    """Final raw benchmark JSONL dosyasının doğru yazıldığını test eder."""
    records = [
        make_record(
            "reasoning_0001_en",
            "reasoning_0001",
            "en",
        ),
        make_record(
            "reasoning_0001_az",
            "reasoning_0001",
            "az",
        ),
    ]

    output_path = tmp_path / "benchmark.jsonl"

    save_final_records(records, output_path)

    assert output_path.exists()

    lines = output_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert len(lines) == 2

    english_record = json.loads(lines[0])
    azerbaijani_record = json.loads(lines[1])

    assert english_record["language"] == "en"
    assert azerbaijani_record["language"] == "az"
    assert english_record["pair_id"] == azerbaijani_record["pair_id"]
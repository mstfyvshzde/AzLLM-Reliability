"""Benchmark builder'ın yükleme, doğrulama ve kaydetme akışını test eder."""

import json
from pathlib import Path

import pytest

from src.data.benchmark_record import BenchmarkRecord
from src.data.build_benchmark import (
    load_records,
    save_records,
)


def test_load_records(tmp_path: Path) -> None:
    """Geçerli JSONL kayıtlarının doğru şekilde yüklendiğini test eder."""
    input_path = tmp_path / "benchmark.jsonl"

    input_path.write_text(
        json.dumps(
            {
                "item_id": "reasoning_001_en",
                "pair_id": "reasoning_001",
                "language": "en",
                "task": "reasoning",
                "question": "Example question",
                "reference_answer": "Example answer",
                "metadata": {"category": "reasoning"}
            }
        )
        + "\n",
        encoding="utf-8"
    )

    records = load_records(input_path)

    assert len(records) == 1
    assert records[0].item_id == "reasoning_001_en"
    assert records[0].pair_id == "reasoning_001"
    assert records[0].language == "en"
    assert records[0].task == "reasoning"


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    """Geçersiz JSONL satırının reddedildiğini test eder."""
    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid benchmark record"):
        load_records(input_path)


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    """Zorunlu alanı eksik olan kaydın reddedildiğini test eder."""
    input_path = tmp_path / "missing.jsonl"

    input_path.write_text(
        json.dumps(
            {
                "item_id": "reasoning_001_en",
                "language": "en",
                "task": "reasoning",
                "question": "Example question",
                "reference_answer": "Example answer"
            }
        )
        + "\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Invalid benchmark record"):
        load_records(input_path)


def test_save_records_by_language(tmp_path: Path) -> None:
    """Kayıtların dillere göre ayrı JSONL dosyalarına yazıldığını test eder."""
    records = [
        BenchmarkRecord(
            item_id="reasoning_001_en",
            pair_id="reasoning_001",
            language="en",
            task="reasoning",
            question="Example question",
            reference_answer="Example answer"
        ),
        BenchmarkRecord(
            item_id="reasoning_001_az",
            pair_id="reasoning_001",
            language="az",
            task="reasoning",
            question="Nümunə sual",
            reference_answer="Nümunə cavab"
        )
    ]

    save_records(records, tmp_path)

    english_output = tmp_path / "en" / "benchmark.jsonl"
    azerbaijani_output = tmp_path / "az" / "benchmark.jsonl"

    assert english_output.exists()
    assert azerbaijani_output.exists()

    english_record = json.loads(
        english_output.read_text(encoding="utf-8").strip()
    )

    azerbaijani_record = json.loads(
        azerbaijani_output.read_text(encoding="utf-8").strip()
    )

    assert english_record["language"] == "en"
    assert english_record["task"] == "reasoning"

    assert azerbaijani_record["language"] == "az"
    assert azerbaijani_record["task"] == "reasoning"

    assert english_record["pair_id"] == azerbaijani_record["pair_id"]
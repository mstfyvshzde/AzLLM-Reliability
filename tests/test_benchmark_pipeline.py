"""Candidate -> reviewed -> raw -> built benchmark akışını uçtan uca test eder."""

import json
from pathlib import Path

from src.data.build_benchmark import (
    load_records,
    save_records,
    validate_records,
)
from src.data.generate_candidates import create_candidate_pair
from src.data.promote_reviewed import (
    save_final_records,
    validate_promoted_records,
)
from src.data.review_candidates import (
    get_approved_records,
    set_pair_review_status,
)


def make_generation_config() -> dict:
    """Pipeline testinde kullanılacak minimal generation config oluşturur."""
    return {
        "generation": {
            "languages": ["en", "az"],
        },
        "identifiers": {
            "pair_id_format": "{task}_{index:04d}",
            "item_id_format": "{pair_id}_{language}",
        },
    }


def make_benchmark_config(tmp_path: Path) -> dict:
    """Pipeline testinde kullanılacak minimal benchmark config oluşturur."""
    return {
        "languages": {
            "source": "en",
            "target": "az",
        },
        "paths": {
            "benchmark_dir": str(tmp_path / "benchmark"),
        },
        "validation": {
            "reject_duplicate_item_ids": True,
            "reject_duplicate_pair_language": True,
        },
        "pairing": {
            "require_both_languages": True,
        },
        "experiment": {
            "overwrite_outputs": False,
        },
    }


def test_full_benchmark_pipeline(tmp_path: Path) -> None:
    """Tek bir EN-AZ pair'in tüm benchmark pipeline'ından geçtiğini test eder."""
    generation_config = make_generation_config()

    candidate_records = create_candidate_pair(
        task="reasoning",
        index=1,
        question_en="What is 2 + 2?",
        question_az="2 + 2 neçə edir?",
        reference_answer_en="4",
        reference_answer_az="4",
        config=generation_config,
    )

    reviewed_records = set_pair_review_status(
        candidate_records,
        pair_id="reasoning_0001",
        status="approved",
    )

    approved_records = get_approved_records(reviewed_records)

    validate_promoted_records(approved_records)

    raw_path = tmp_path / "raw" / "benchmark.jsonl"
    save_final_records(approved_records, raw_path)

    loaded_records = load_records(raw_path)

    benchmark_config = make_benchmark_config(tmp_path)

    validate_records(
        loaded_records,
        benchmark_config,
        {"reasoning"},
    )

    output_dir = Path(
        benchmark_config["paths"]["benchmark_dir"]
    )

    save_records(loaded_records, output_dir)

    english_output = output_dir / "en" / "benchmark.jsonl"
    azerbaijani_output = output_dir / "az" / "benchmark.jsonl"

    assert english_output.exists()
    assert azerbaijani_output.exists()

    english_record = json.loads(
        english_output.read_text(encoding="utf-8").strip()
    )

    azerbaijani_record = json.loads(
        azerbaijani_output.read_text(encoding="utf-8").strip()
    )

    assert english_record["pair_id"] == "reasoning_0001"
    assert azerbaijani_record["pair_id"] == "reasoning_0001"

    assert english_record["language"] == "en"
    assert azerbaijani_record["language"] == "az"

    assert english_record["task"] == "reasoning"
    assert azerbaijani_record["task"] == "reasoning"
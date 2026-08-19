"""Candidate generation yapılandırma ve pair üretim davranışlarını test eder."""

from pathlib import Path

import pytest

from src.data.generate_candidates import (
    create_candidate_pair,
    load_generation_config,
)


def make_generation_config() -> dict:
    """Testlerde kullanılacak minimal generation config oluşturur."""
    return {
        "generation": {
            "languages": ["en", "az"]
        },
        "identifiers": {
            "pair_id_format": "{task}_{index:04d}",
            "item_id_format": "{pair_id}_{language}"
        }
    }


def make_task_specification() -> dict:
    """Testlerde kullanılacak minimal reasoning task specification oluşturur."""
    return {
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


def test_load_generation_config(tmp_path: Path) -> None:
    """Geçerli generation config dosyasının yüklendiğini test eder."""
    config_path = tmp_path / "generation.yaml"

    config_path.write_text(
        """
generation:
  languages:
    - en
    - az

identifiers:
  pair_id_format: "{task}_{index:04d}"
  item_id_format: "{pair_id}_{language}"
""",
        encoding="utf-8"
    )

    config = load_generation_config(config_path)

    assert "generation" in config
    assert "identifiers" in config


def test_missing_generation_section_is_rejected(
    tmp_path: Path,
) -> None:
    """generation bölümü olmayan config dosyasının reddedildiğini test eder."""
    config_path = tmp_path / "generation.yaml"

    config_path.write_text(
        "project: test\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="generation"):
        load_generation_config(config_path)


def test_create_candidate_pair() -> None:
    """Tek semantic pair için EN ve AZ kayıtlarının üretildiğini test eder."""
    config = make_generation_config()

    records = create_candidate_pair(
        task="reasoning",
        index=1,
        question_en="What is 2 + 2?",
        question_az="2 + 2 neçə edir?",
        reference_answer_en="4",
        reference_answer_az="4",
        category="arithmetic_reasoning",
        difficulty="easy",
        config=config,
        task_specification=make_task_specification(),
    )

    assert len(records) == 2

    english_record = records[0]
    azerbaijani_record = records[1]

    assert english_record.pair_id == "reasoning_0001"
    assert azerbaijani_record.pair_id == "reasoning_0001"

    assert english_record.item_id == "reasoning_0001_en"
    assert azerbaijani_record.item_id == "reasoning_0001_az"

    assert english_record.language == "en"
    assert azerbaijani_record.language == "az"

    assert english_record.task == "reasoning"
    assert azerbaijani_record.task == "reasoning"


def test_candidate_pair_review_status_is_pending() -> None:
    """Yeni candidate kayıtlarının pending review status aldığını test eder."""
    config = make_generation_config()

    records = create_candidate_pair(
        task="reasoning",
        index=1,
        question_en="Example question",
        question_az="Nümunə sual",
        reference_answer_en="Example answer",
        reference_answer_az="Nümunə cavab",
        category="arithmetic_reasoning",
        difficulty="easy",
        config=config,
        task_specification=make_task_specification(),
    )

    for record in records:
        assert record.metadata["review_status"] == "pending"


def test_invalid_category_is_rejected() -> None:
    """Tanımlanmamış category değerinin candidate aşamasında reddedildiğini test eder."""
    config = make_generation_config()

    with pytest.raises(ValueError, match="Unsupported category"):
        create_candidate_pair(
            task="reasoning",
            index=1,
            question_en="What is 2 + 2?",
            question_az="2 + 2 neçə edir?",
            reference_answer_en="4",
            reference_answer_az="4",
            category="unknown_reasoning",
            difficulty="easy",
            config=config,
            task_specification=make_task_specification(),
        )


def test_invalid_difficulty_is_rejected() -> None:
    """Geçersiz difficulty değerinin candidate aşamasında reddedildiğini test eder."""
    config = make_generation_config()

    with pytest.raises(ValueError, match="Unsupported difficulty"):
        create_candidate_pair(
            task="reasoning",
            index=1,
            question_en="What is 2 + 2?",
            question_az="2 + 2 neçə edir?",
            reference_answer_en="4",
            reference_answer_az="4",
            category="arithmetic_reasoning",
            difficulty="extreme",
            config=config,
            task_specification=make_task_specification(),
        )
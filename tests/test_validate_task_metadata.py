"""Task-specific metadata validation yardımcılarını test eder."""

from pathlib import Path

import pytest
import yaml

from src.data.benchmark_record import BenchmarkRecord
from src.data.validate_task_metadata import (
    load_task_specifications,
    validate_category,
    validate_difficulty,
    validate_metadata_constraints,
    validate_required_metadata,
    validate_task_metadata,
)


def make_record(
    *,
    task: str = "reasoning",
    category: str = "logical_reasoning",
    difficulty: str = "easy",
    review_status: str = "pending",
    is_answerable: bool = True,
) -> BenchmarkRecord:
    """Testler için örnek BenchmarkRecord oluşturur."""

    return BenchmarkRecord(
        item_id=f"{task}_001_en",
        pair_id=f"{task}_001",
        language="en",
        task=task,
        question="What is the answer?",
        reference_answer="4",
        metadata={
            "category": category,
            "difficulty": difficulty,
            "review_status": review_status,
            "is_answerable": is_answerable,
        },
    )


def make_reasoning_specification() -> dict:
    """Reasoning task için örnek specification oluşturur."""

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
                "is_answerable",
            ],
            "constraints": {
                "is_answerable": True,
            },
        },
    }


def make_unanswerable_specification() -> dict:
    """Unanswerable task için örnek specification oluşturur."""

    return {
        "task": {
            "name": "unanswerable",
        },
        "categories": {
            "missing_information": {
                "enabled": True,
            },
            "ambiguous_reference": {
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
                "is_answerable",
            ],
            "constraints": {
                "is_answerable": False,
            },
        },
    }


def test_load_task_specifications(
    tmp_path: Path,
) -> None:
    """Geçerli task specification YAML dosyasının yüklendiğini test eder."""

    specification_path = (
        tmp_path
        / "reasoning.yaml"
    )

    specification = make_reasoning_specification()

    with specification_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            specification,
            file,
        )

    loaded = load_task_specifications(
        specification_path
    )

    assert loaded[
        "task"
    ][
        "name"
    ] == "reasoning"


def test_load_task_specifications_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Mevcut olmayan task specification dosyasını reddeder."""

    with pytest.raises(
        FileNotFoundError,
        match="Task specification not found",
    ):
        load_task_specifications(
            tmp_path
            / "missing.yaml"
        )


def test_load_task_specifications_rejects_non_mapping(
    tmp_path: Path,
) -> None:
    """Root YAML mapping değilse reddedildiğini test eder."""

    specification_path = (
        tmp_path
        / "invalid.yaml"
    )

    specification_path.write_text(
        "- reasoning\n- task\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Task specification must contain a YAML mapping",
    ):
        load_task_specifications(
            specification_path
        )


def test_load_task_specifications_rejects_missing_task_section(
    tmp_path: Path,
) -> None:
    """Task bölümü olmayan specification dosyasını reddeder."""

    specification_path = (
        tmp_path
        / "invalid.yaml"
    )

    specification_path.write_text(
        "metadata: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must contain a 'task' section",
    ):
        load_task_specifications(
            specification_path
        )


def test_validate_required_metadata() -> None:
    """Tüm required metadata alanları mevcutsa validation geçer."""

    record = make_record()

    validate_required_metadata(
        record,
        make_reasoning_specification(),
    )


def test_validate_required_metadata_rejects_missing_field() -> None:
    """Required metadata alanı eksikse reddedildiğini test eder."""

    record = make_record()

    metadata = dict(
        record.metadata
    )

    metadata.pop(
        "is_answerable"
    )

    record = BenchmarkRecord(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        question=record.question,
        reference_answer=record.reference_answer,
        metadata=metadata,
    )

    with pytest.raises(
        ValueError,
        match="Missing required metadata 'is_answerable'",
    ):
        validate_required_metadata(
            record,
            make_reasoning_specification(),
        )


def test_validate_required_metadata_rejects_empty_string() -> None:
    """Required string metadata boşsa reddedildiğini test eder."""

    record = make_record(
        review_status="   "
    )

    with pytest.raises(
        ValueError,
        match="Missing required metadata 'review_status'",
    ):
        validate_required_metadata(
            record,
            make_reasoning_specification(),
        )


def test_validate_category() -> None:
    """Enabled category değerinin kabul edildiğini test eder."""

    record = make_record(
        category="logical_reasoning"
    )

    validate_category(
        record,
        make_reasoning_specification(),
    )


def test_validate_category_rejects_unknown_category() -> None:
    """Specification içinde olmayan category değerini reddeder."""

    record = make_record(
        category="unknown_reasoning"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported category",
    ):
        validate_category(
            record,
            make_reasoning_specification(),
        )


def test_validate_category_rejects_disabled_category() -> None:
    """Disabled category değerini reddeder."""

    specification = make_reasoning_specification()

    specification[
        "categories"
    ][
        "logical_reasoning"
    ][
        "enabled"
    ] = False

    record = make_record(
        category="logical_reasoning"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported category",
    ):
        validate_category(
            record,
            specification,
        )


def test_validate_difficulty() -> None:
    """Geçerli difficulty değerinin kabul edildiğini test eder."""

    record = make_record(
        difficulty="medium"
    )

    validate_difficulty(
        record,
        make_reasoning_specification(),
    )


def test_validate_difficulty_rejects_unknown_level() -> None:
    """Tanımsız difficulty seviyesini reddeder."""

    record = make_record(
        difficulty="expert"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported difficulty",
    ):
        validate_difficulty(
            record,
            make_reasoning_specification(),
        )


def test_validate_metadata_constraints() -> None:
    """Reasoning constraint değerinin doğru olduğunda validation geçer."""

    record = make_record(
        is_answerable=True
    )

    validate_metadata_constraints(
        record,
        make_reasoning_specification(),
    )


def test_validate_metadata_constraints_rejects_wrong_value() -> None:
    """Constraint ile uyuşmayan metadata değerini reddeder."""

    record = make_record(
        is_answerable=False
    )

    with pytest.raises(
        ValueError,
        match="Invalid metadata constraint",
    ):
        validate_metadata_constraints(
            record,
            make_reasoning_specification(),
        )


def test_validate_metadata_constraints_rejects_missing_field() -> None:
    """Constraint alanı metadata içinde yoksa reddeder."""

    record = make_record()

    metadata = dict(
        record.metadata
    )

    metadata.pop(
        "is_answerable"
    )

    record = BenchmarkRecord(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        question=record.question,
        reference_answer=record.reference_answer,
        metadata=metadata,
    )

    with pytest.raises(
        ValueError,
        match="Missing constrained metadata 'is_answerable'",
    ):
        validate_metadata_constraints(
            record,
            make_reasoning_specification(),
        )


def test_validate_metadata_constraints_rejects_non_mapping() -> None:
    """metadata.constraints mapping değilse reddeder."""

    specification = make_reasoning_specification()

    specification[
        "metadata"
    ][
        "constraints"
    ] = [
        "is_answerable"
    ]

    record = make_record()

    with pytest.raises(
        ValueError,
        match="metadata.constraints must be a mapping",
    ):
        validate_metadata_constraints(
            record,
            specification,
        )


def test_validate_reasoning_metadata() -> None:
    """Geçerli reasoning record'un tamamının doğrulandığını test eder."""

    record = make_record(
        task="reasoning",
        category="logical_reasoning",
        difficulty="easy",
        is_answerable=True,
    )

    validate_task_metadata(
        record,
        make_reasoning_specification(),
    )


def test_validate_reasoning_rejects_false_is_answerable() -> None:
    """Reasoning task için is_answerable=False değerini reddeder."""

    record = make_record(
        task="reasoning",
        category="logical_reasoning",
        is_answerable=False,
    )

    with pytest.raises(
        ValueError,
        match="Invalid metadata constraint",
    ):
        validate_task_metadata(
            record,
            make_reasoning_specification(),
        )


def test_validate_unanswerable_metadata() -> None:
    """Geçerli unanswerable record'un tamamının doğrulandığını test eder."""

    record = make_record(
        task="unanswerable",
        category="missing_information",
        difficulty="medium",
        is_answerable=False,
    )

    validate_task_metadata(
        record,
        make_unanswerable_specification(),
    )


def test_validate_unanswerable_rejects_true_is_answerable() -> None:
    """Unanswerable task için is_answerable=True değerini reddeder."""

    record = make_record(
        task="unanswerable",
        category="missing_information",
        difficulty="medium",
        is_answerable=True,
    )

    with pytest.raises(
        ValueError,
        match="Invalid metadata constraint",
    ):
        validate_task_metadata(
            record,
            make_unanswerable_specification(),
        )


def test_validate_task_metadata_rejects_task_mismatch() -> None:
    """Record task ve specification task farklıysa reddeder."""

    record = make_record(
        task="reasoning"
    )

    specification = make_unanswerable_specification()

    with pytest.raises(
        ValueError,
        match="does not match task specification",
    ):
        validate_task_metadata(
            record,
            specification,
        )
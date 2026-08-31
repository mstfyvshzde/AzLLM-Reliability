"""Candidate benchmark validation yardımcılarını test eder."""

from pathlib import Path

import pytest
import yaml

from src.data.benchmark_record import BenchmarkRecord
from src.data.validate_candidates import (
    discover_candidate_files,
    get_task_names,
    load_task_specification_for_task,
    validate_all_candidates,
    validate_candidate_file,
    validate_candidate_records,
    validate_task_records,
)


def make_record(
    *,
    item_id: str,
    pair_id: str,
    language: str,
    task: str = "reasoning",
    category: str = "logical_reasoning",
    difficulty: str = "easy",
    is_answerable: bool = True,
) -> BenchmarkRecord:
    """Candidate validation testleri için örnek BenchmarkRecord oluşturur."""

    return BenchmarkRecord(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        question="What is the answer?",
        reference_answer="4",
        metadata={
            "category": category,
            "difficulty": difficulty,
            "review_status": "pending",
            "is_answerable": is_answerable,
        },
    )


def write_task_specification(
    specifications_dir: Path,
    *,
    task: str,
    category: str,
    is_answerable: bool,
) -> Path:
    """Test için minimal task specification YAML dosyası oluşturur."""

    specifications_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    specification_path = (
        specifications_dir
        / f"{task}.yaml"
    )

    specification = {
        "task": {
            "name": task,
        },
        "categories": {
            category: {
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
                "is_answerable": is_answerable,
            },
        },
    }

    with specification_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            specification,
            file,
        )

    return specification_path


def write_records(
    output_path: Path,
    records: list[BenchmarkRecord],
) -> None:
    """BenchmarkRecord listesini JSONL test dosyasına yazar."""

    import json

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            json.dump(
                record.to_dict(),
                file,
                ensure_ascii=False,
            )

            file.write(
                "\n"
            )


def make_reasoning_pair() -> list[BenchmarkRecord]:
    """Geçerli EN-AZ reasoning pair oluşturur."""

    return [
        make_record(
            item_id="reasoning_0001_en",
            pair_id="reasoning_0001",
            language="en",
        ),
        make_record(
            item_id="reasoning_0001_az",
            pair_id="reasoning_0001",
            language="az",
        ),
    ]


def test_discover_candidate_files(
    tmp_path: Path,
) -> None:
    """Candidate JSONL dosyalarının alfabetik bulunduğunu test eder."""

    candidates_dir = (
        tmp_path
        / "candidates"
    )

    candidates_dir.mkdir()

    (
        candidates_dir
        / "reasoning.jsonl"
    ).write_text(
        "",
        encoding="utf-8",
    )

    (
        candidates_dir
        / "factual_knowledge.jsonl"
    ).write_text(
        "",
        encoding="utf-8",
    )

    (
        candidates_dir
        / "notes.txt"
    ).write_text(
        "",
        encoding="utf-8",
    )

    files = discover_candidate_files(
        candidates_dir
    )

    assert [
        path.name
        for path in files
    ] == [
        "factual_knowledge.jsonl",
        "reasoning.jsonl",
    ]


def test_discover_candidate_files_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """Mevcut olmayan candidate klasörünü reddeder."""

    with pytest.raises(
        FileNotFoundError,
        match="Candidates directory not found",
    ):
        discover_candidate_files(
            tmp_path
            / "missing"
        )


def test_discover_candidate_files_rejects_non_directory(
    tmp_path: Path,
) -> None:
    """Candidate path klasör değilse reddeder."""

    path = (
        tmp_path
        / "candidates"
    )

    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Candidates path is not a directory",
    ):
        discover_candidate_files(
            path
        )


def test_discover_candidate_files_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    """JSONL candidate içermeyen klasörü reddeder."""

    candidates_dir = (
        tmp_path
        / "candidates"
    )

    candidates_dir.mkdir()

    with pytest.raises(
        ValueError,
        match="No candidate JSONL files found",
    ):
        discover_candidate_files(
            candidates_dir
        )


def test_get_task_names() -> None:
    """Record listesindeki unique task adlarını döndürür."""

    records = make_reasoning_pair()

    assert get_task_names(
        records
    ) == {
        "reasoning",
    }


def test_get_task_names_multiple_tasks() -> None:
    """Birden fazla task'ın doğru toplandığını test eder."""

    records = make_reasoning_pair()

    records.extend(
        [
            make_record(
                item_id="unanswerable_0001_en",
                pair_id="unanswerable_0001",
                language="en",
                task="unanswerable",
                category="missing_information",
                is_answerable=False,
            ),
            make_record(
                item_id="unanswerable_0001_az",
                pair_id="unanswerable_0001",
                language="az",
                task="unanswerable",
                category="missing_information",
                is_answerable=False,
            ),
        ]
    )

    assert get_task_names(
        records
    ) == {
        "reasoning",
        "unanswerable",
    }


def test_get_task_names_rejects_empty_records() -> None:
    """Boş record listesini reddeder."""

    with pytest.raises(
        ValueError,
        match="Candidate records contain no tasks",
    ):
        get_task_names(
            []
        )


def test_load_task_specification_for_task(
    tmp_path: Path,
) -> None:
    """Task adına göre doğru specification dosyasını yükler."""

    specifications_dir = (
        tmp_path
        / "tasks"
    )

    write_task_specification(
        specifications_dir,
        task="reasoning",
        category="logical_reasoning",
        is_answerable=True,
    )

    specification = load_task_specification_for_task(
        task="reasoning",
        specifications_dir=specifications_dir,
    )

    assert specification[
        "task"
    ][
        "name"
    ] == "reasoning"


def test_validate_task_records() -> None:
    """Record'ların kendi task specification'larına göre doğrulandığını test eder."""

    records = make_reasoning_pair()

    specifications = {
        "reasoning": {
            "task": {
                "name": "reasoning",
            },
            "categories": {
                "logical_reasoning": {
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
    }

    validate_task_records(
        records=records,
        task_specifications=specifications,
    )


def test_validate_task_records_rejects_missing_specification() -> None:
    """Record task'ı için specification yoksa reddeder."""

    records = make_reasoning_pair()

    with pytest.raises(
        ValueError,
        match="No task specification loaded for task 'reasoning'",
    ):
        validate_task_records(
            records=records,
            task_specifications={},
        )


def test_validate_candidate_records(
    tmp_path: Path,
) -> None:
    """Geçerli candidate pair'in tüm validation zincirinden geçtiğini test eder."""

    specifications_dir = (
        tmp_path
        / "tasks"
    )

    write_task_specification(
        specifications_dir,
        task="reasoning",
        category="logical_reasoning",
        is_answerable=True,
    )

    validate_candidate_records(
        records=make_reasoning_pair(),
        specifications_dir=specifications_dir,
        required_languages={
            "en",
            "az",
        },
    )


def test_validate_candidate_records_rejects_empty_records(
    tmp_path: Path,
) -> None:
    """Boş candidate record listesini reddeder."""

    with pytest.raises(
        ValueError,
        match="Candidate records cannot be empty",
    ):
        validate_candidate_records(
            records=[],
            specifications_dir=tmp_path,
            required_languages={
                "en",
                "az",
            },
        )


def test_validate_candidate_records_rejects_duplicate_item_id(
    tmp_path: Path,
) -> None:
    """Duplicate item_id değerlerini reddeder."""

    specifications_dir = (
        tmp_path
        / "tasks"
    )

    write_task_specification(
        specifications_dir,
        task="reasoning",
        category="logical_reasoning",
        is_answerable=True,
    )

    records = [
        make_record(
            item_id="duplicate",
            pair_id="pair_1",
            language="en",
        ),
        make_record(
            item_id="duplicate",
            pair_id="pair_1",
            language="az",
        ),
    ]

    with pytest.raises(
        ValueError,
    ):
        validate_candidate_records(
            records=records,
            specifications_dir=specifications_dir,
            required_languages={
                "en",
                "az",
            },
        )


def test_validate_candidate_records_rejects_duplicate_pair_language(
    tmp_path: Path,
) -> None:
    """Aynı pair içinde aynı language tekrarını reddeder."""

    specifications_dir = (
        tmp_path
        / "tasks"
    )

    write_task_specification(
        specifications_dir,
        task="reasoning",
        category="logical_reasoning",
        is_answerable=True,
    )

    records = [
        make_record(
            item_id="reasoning_0001_en_a",
            pair_id="reasoning_0001",
            language="en",
        ),
        make_record(
            item_id="reasoning_0001_en_b",
            pair_id="reasoning_0001",
            language="en",
        ),
    ]

    with pytest.raises(
        ValueError,
    ):
        validate_candidate_records(
            records=records,
            specifications_dir=specifications_dir,
            required_languages={
                "en",
                "az",
            },
        )


def test_validate_candidate_records_rejects_incomplete_pair(
    tmp_path: Path,
) -> None:
    """EN-AZ pair incomplete ise reddeder."""

    specifications_dir = (
        tmp_path
        / "tasks"
    )

    write_task_specification(
        specifications_dir,
        task="reasoning",
        category="logical_reasoning",
        is_answerable=True,
    )

    records = [
        make_record(
            item_id="reasoning_0001_en",
            pair_id="reasoning_0001",
            language="en",
        )
    ]

    with pytest.raises(
        ValueError,
    ):
        validate_candidate_records(
            records=records,
            specifications_dir=specifications_dir,
            required_languages={
                "en",
                "az",
            },
        )


def test_validate_candidate_records_rejects_pair_task_mismatch(
    tmp_path: Path,
) -> None:
    """Aynı pair içinde farklı task değerlerini reddeder."""

    specifications_dir = (
        tmp_path
        / "tasks"
    )

    records = [
        make_record(
            item_id="pair_1_en",
            pair_id="pair_1",
            language="en",
            task="reasoning",
        ),
        make_record(
            item_id="pair_1_az",
            pair_id="pair_1",
            language="az",
            task="unanswerable",
            category="missing_information",
            is_answerable=False,
        ),
    ]

    with pytest.raises(
        ValueError,
    ):
        validate_candidate_records(
            records=records,
            specifications_dir=specifications_dir,
            required_languages={
                "en",
                "az",
            },
        )


def test_validate_candidate_records_rejects_wrong_constraint(
    tmp_path: Path,
) -> None:
    """Task is_answerable constraint'ini ihlal eden record'u reddeder."""

    specifications_dir = (
        tmp_path
        / "tasks"
    )

    write_task_specification(
        specifications_dir,
        task="reasoning",
        category="logical_reasoning",
        is_answerable=True,
    )

    records = [
        make_record(
            item_id="reasoning_0001_en",
            pair_id="reasoning_0001",
            language="en",
            is_answerable=False,
        ),
        make_record(
            item_id="reasoning_0001_az",
            pair_id="reasoning_0001",
            language="az",
            is_answerable=False,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Invalid metadata constraint",
    ):
        validate_candidate_records(
            records=records,
            specifications_dir=specifications_dir,
            required_languages={
                "en",
                "az",
            },
        )


def test_validate_candidate_file(
    tmp_path: Path,
) -> None:
    """Tek candidate dosyasının validate edilip record sayısını döndürdüğünü test eder."""

    specifications_dir = (
        tmp_path
        / "tasks"
    )

    candidates_dir = (
        tmp_path
        / "candidates"
    )

    write_task_specification(
        specifications_dir,
        task="reasoning",
        category="logical_reasoning",
        is_answerable=True,
    )

    candidate_path = (
        candidates_dir
        / "reasoning.jsonl"
    )

    records = make_reasoning_pair()

    write_records(
        candidate_path,
        records,
    )

    record_count = validate_candidate_file(
        candidate_path=candidate_path,
        specifications_dir=specifications_dir,
        required_languages={
            "en",
            "az",
        },
    )

    assert record_count == 2


def test_validate_all_candidates(
    tmp_path: Path,
) -> None:
    """Candidate klasöründeki tüm task dosyalarını topluca doğrular."""

    specifications_dir = (
        tmp_path
        / "tasks"
    )

    candidates_dir = (
        tmp_path
        / "candidates"
    )

    write_task_specification(
        specifications_dir,
        task="reasoning",
        category="logical_reasoning",
        is_answerable=True,
    )

    write_task_specification(
        specifications_dir,
        task="unanswerable",
        category="missing_information",
        is_answerable=False,
    )

    reasoning_records = make_reasoning_pair()

    unanswerable_records = [
        make_record(
            item_id="unanswerable_0001_en",
            pair_id="unanswerable_0001",
            language="en",
            task="unanswerable",
            category="missing_information",
            is_answerable=False,
        ),
        make_record(
            item_id="unanswerable_0001_az",
            pair_id="unanswerable_0001",
            language="az",
            task="unanswerable",
            category="missing_information",
            is_answerable=False,
        ),
    ]

    write_records(
        candidates_dir
        / "reasoning.jsonl",
        reasoning_records,
    )

    write_records(
        candidates_dir
        / "unanswerable.jsonl",
        unanswerable_records,
    )

    results = validate_all_candidates(
        candidates_dir=candidates_dir,
        specifications_dir=specifications_dir,
        required_languages={
            "en",
            "az",
        },
    )

    assert results == {
        "reasoning.jsonl": 2,
        "unanswerable.jsonl": 2,
    }
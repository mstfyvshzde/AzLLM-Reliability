"""Candidate content preparation testleri."""

from pathlib import Path

import pytest

from src.data.generate_benchmark_plan import BenchmarkPlanRecord
from src.data.prepare_candidate_content import (
    CandidateGenerationRequest,
    build_generation_instruction,
    prepare_generation_requests,
    save_generation_batches,
    save_generation_requests,
    split_into_batches,
    summarize_generation_requests,
)


def make_plan_record(
    *,
    pair_id: str = "reasoning_0001",
    task: str = "reasoning",
    category: str = "logical_reasoning",
    difficulty: str = "medium",
    is_answerable: bool = True,
) -> BenchmarkPlanRecord:
    """Testlerde kullanılacak benchmark plan kaydını oluşturur."""

    return BenchmarkPlanRecord(
        pair_id=pair_id,
        task=task,
        category=category,
        difficulty=difficulty,
        source_language="en",
        target_language="az",
        is_answerable=is_answerable,
    )


def make_request(
    *,
    pair_id: str = "reasoning_0001",
) -> CandidateGenerationRequest:
    """Testlerde kullanılacak generation request oluşturur."""

    return CandidateGenerationRequest(
        pair_id=pair_id,
        task="reasoning",
        category="logical_reasoning",
        difficulty="medium",
        source_language="en",
        target_language="az",
        is_answerable=True,
        instruction="Test instruction",
    )


def test_build_generation_instruction() -> None:
    record = make_plan_record()

    instruction = build_generation_instruction(
        record
    )

    assert "Task: reasoning" in instruction
    assert "Category: logical_reasoning" in instruction
    assert "Difficulty: medium" in instruction
    assert "English-Azerbaijani pair" in instruction


def test_build_generation_instruction_for_unanswerable() -> None:
    record = make_plan_record(
        pair_id="unanswerable_0001",
        task="unanswerable",
        category="missing_information",
        difficulty="hard",
        is_answerable=False,
    )

    instruction = build_generation_instruction(
        record
    )

    assert "must not have a definitive answer" in instruction
    assert "explicitly abstain" in instruction


def test_build_generation_instruction_rejects_unknown_task() -> None:
    record = make_plan_record(
        task="unknown_task",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported generation task",
    ):
        build_generation_instruction(
            record
        )


def test_build_generation_instruction_rejects_unknown_category() -> None:
    record = make_plan_record(
        category="unknown_category",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported generation category",
    ):
        build_generation_instruction(
            record
        )


def test_build_generation_instruction_rejects_unknown_difficulty() -> None:
    record = make_plan_record(
        difficulty="extreme",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported difficulty",
    ):
        build_generation_instruction(
            record
        )


def test_prepare_generation_requests() -> None:
    plan = [
        make_plan_record(
            pair_id="reasoning_0001",
        ),
        make_plan_record(
            pair_id="reasoning_0002",
        ),
    ]

    requests = prepare_generation_requests(
        plan
    )

    assert len(requests) == 2

    assert requests[0].pair_id == "reasoning_0001"
    assert requests[1].pair_id == "reasoning_0002"

    assert requests[0].source_language == "en"
    assert requests[0].target_language == "az"


def test_split_into_batches() -> None:
    records = [
        make_request(
            pair_id=f"reasoning_{index:04d}"
        )
        for index in range(1, 11)
    ]

    batches = split_into_batches(
        records,
        batch_size=4,
    )

    assert len(batches) == 3

    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 2


def test_split_into_batches_rejects_zero_batch_size() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        split_into_batches(
            [],
            batch_size=0,
        )


def test_split_into_batches_rejects_negative_batch_size() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        split_into_batches(
            [],
            batch_size=-1,
        )


def test_summarize_generation_requests() -> None:
    plan = [
        make_plan_record(
            pair_id="reasoning_0001",
            task="reasoning",
            category="logical_reasoning",
            difficulty="easy",
            is_answerable=True,
        ),
        make_plan_record(
            pair_id="reasoning_0002",
            task="reasoning",
            category="logical_reasoning",
            difficulty="medium",
            is_answerable=True,
        ),
        make_plan_record(
            pair_id="unanswerable_0001",
            task="unanswerable",
            category="missing_information",
            difficulty="hard",
            is_answerable=False,
        ),
    ]

    requests = prepare_generation_requests(
        plan
    )

    summary = summarize_generation_requests(
        requests
    )

    assert summary == {
        "total_requests": 3,
        "by_task": {
            "reasoning": 2,
            "unanswerable": 1,
        },
        "by_difficulty": {
            "easy": 1,
            "medium": 1,
            "hard": 1,
        },
        "answerable": 2,
        "unanswerable": 1,
    }


def test_save_generation_requests(
    tmp_path: Path,
) -> None:
    records = [
        make_request(
            pair_id="reasoning_0001",
        ),
        make_request(
            pair_id="reasoning_0002",
        ),
    ]

    output_path = (
        tmp_path
        / "requests.jsonl"
    )

    save_generation_requests(
        records,
        output_path,
    )

    assert output_path.exists()

    lines = output_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert len(lines) == 2


def test_save_generation_requests_rejects_existing_file(
    tmp_path: Path,
) -> None:
    records = [
        make_request(),
    ]

    output_path = (
        tmp_path
        / "requests.jsonl"
    )

    save_generation_requests(
        records,
        output_path,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        save_generation_requests(
            records,
            output_path,
        )


def test_save_generation_batches(
    tmp_path: Path,
) -> None:
    records = [
        make_request(
            pair_id=f"reasoning_{index:04d}"
        )
        for index in range(1, 6)
    ]

    batches = split_into_batches(
        records,
        batch_size=2,
    )

    output_dir = (
        tmp_path
        / "batches"
    )

    save_generation_batches(
        batches,
        output_dir,
    )

    files = sorted(
        output_dir.glob(
            "batch_*.jsonl"
        )
    )

    assert len(files) == 3

    assert files[0].name == "batch_001.jsonl"
    assert files[1].name == "batch_002.jsonl"
    assert files[2].name == "batch_003.jsonl"


def test_save_generation_batches_rejects_existing_file(
    tmp_path: Path,
) -> None:
    records = [
        make_request(),
    ]

    batches = split_into_batches(
        records,
        batch_size=1,
    )

    output_dir = (
        tmp_path
        / "batches"
    )

    save_generation_batches(
        batches,
        output_dir,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        save_generation_batches(
            batches,
            output_dir,
        )


def test_candidate_generation_request_to_dict() -> None:
    request = make_request()

    assert request.to_dict() == {
        "pair_id": "reasoning_0001",
        "task": "reasoning",
        "category": "logical_reasoning",
        "difficulty": "medium",
        "source_language": "en",
        "target_language": "az",
        "is_answerable": True,
        "instruction": "Test instruction",
    }
"""Final benchmark candidate generation testleri."""

import json
from pathlib import Path

import pytest

from src.data.generate_benchmark_plan import BenchmarkPlanRecord
from src.data.generate_final_candidates import (
    CandidateContentRecord,
    build_candidate_pair,
    generate_final_candidates,
    load_candidate_content,
    load_generation_plan,
    save_final_candidates,
    summarize_final_candidates,
    validate_generated_candidate_ids,
    validate_plan_content_alignment,
    validate_unique_content_pair_ids,
    validate_unique_plan_pair_ids,
)


def make_plan_record(
    *,
    pair_id: str = "reasoning_0001",
    task: str = "reasoning",
    category: str = "logical_reasoning",
    difficulty: str = "medium",
    is_answerable: bool = True,
) -> BenchmarkPlanRecord:
    """Testlerde kullanılacak plan kaydını oluşturur."""

    return BenchmarkPlanRecord(
        pair_id=pair_id,
        task=task,
        category=category,
        difficulty=difficulty,
        source_language="en",
        target_language="az",
        is_answerable=is_answerable,
    )


def make_content_record(
    *,
    pair_id: str = "reasoning_0001",
    source_question: str = "What is 2 + 2?",
    source_reference_answer: str = "4",
    target_question: str = "2 + 2 neçə edir?",
    target_reference_answer: str = "4",
) -> CandidateContentRecord:
    """Testlerde kullanılacak content kaydını oluşturur."""

    return CandidateContentRecord(
        pair_id=pair_id,
        source_question=source_question,
        source_reference_answer=source_reference_answer,
        target_question=target_question,
        target_reference_answer=target_reference_answer,
    )


def test_validate_unique_plan_pair_ids() -> None:
    plan = [
        make_plan_record(
            pair_id="reasoning_0001",
        ),
        make_plan_record(
            pair_id="reasoning_0002",
        ),
    ]

    validate_unique_plan_pair_ids(plan)


def test_validate_unique_plan_pair_ids_rejects_duplicate() -> None:
    plan = [
        make_plan_record(),
        make_plan_record(),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate pair_id in generation plan",
    ):
        validate_unique_plan_pair_ids(plan)


def test_validate_unique_content_pair_ids() -> None:
    content = [
        make_content_record(
            pair_id="reasoning_0001",
        ),
        make_content_record(
            pair_id="reasoning_0002",
        ),
    ]

    validate_unique_content_pair_ids(content)


def test_validate_unique_content_pair_ids_rejects_duplicate() -> None:
    content = [
        make_content_record(),
        make_content_record(),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate pair_id in candidate content",
    ):
        validate_unique_content_pair_ids(content)


def test_validate_plan_content_alignment() -> None:
    plan = [
        make_plan_record(
            pair_id="reasoning_0001",
        ),
        make_plan_record(
            pair_id="reasoning_0002",
        ),
    ]

    content = [
        make_content_record(
            pair_id="reasoning_0001",
        ),
        make_content_record(
            pair_id="reasoning_0002",
        ),
    ]

    validate_plan_content_alignment(
        plan,
        content,
    )


def test_validate_plan_content_alignment_rejects_missing_content() -> None:
    plan = [
        make_plan_record(
            pair_id="reasoning_0001",
        ),
        make_plan_record(
            pair_id="reasoning_0002",
        ),
    ]

    content = [
        make_content_record(
            pair_id="reasoning_0001",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="missing planned pairs",
    ):
        validate_plan_content_alignment(
            plan,
            content,
        )


def test_validate_plan_content_alignment_rejects_unexpected_content() -> None:
    plan = [
        make_plan_record(
            pair_id="reasoning_0001",
        ),
    ]

    content = [
        make_content_record(
            pair_id="reasoning_0001",
        ),
        make_content_record(
            pair_id="reasoning_9999",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="unexpected pairs",
    ):
        validate_plan_content_alignment(
            plan,
            content,
        )


def test_build_candidate_pair() -> None:
    plan_record = make_plan_record()

    content_record = make_content_record()

    source, target = build_candidate_pair(
        plan_record,
        content_record,
    )

    assert source.item_id == "reasoning_0001_en"
    assert target.item_id == "reasoning_0001_az"

    assert source.language == "en"
    assert target.language == "az"

    assert source.task == "reasoning"
    assert target.task == "reasoning"

    assert source.question == "What is 2 + 2?"
    assert target.question == "2 + 2 neçə edir?"

    assert source.reference_answer == "4"
    assert target.reference_answer == "4"

    assert source.metadata["category"] == "logical_reasoning"
    assert source.metadata["difficulty"] == "medium"
    assert source.metadata["review_status"] == "pending"
    assert source.metadata["is_answerable"] is True

    assert target.metadata == source.metadata


def test_build_candidate_pair_rejects_pair_id_mismatch() -> None:
    plan_record = make_plan_record(
        pair_id="reasoning_0001",
    )

    content_record = make_content_record(
        pair_id="reasoning_0002",
    )

    with pytest.raises(
        ValueError,
        match="pair_id mismatch",
    ):
        build_candidate_pair(
            plan_record,
            content_record,
        )


def test_generate_final_candidates() -> None:
    plan = [
        make_plan_record(
            pair_id="reasoning_0001",
        ),
        make_plan_record(
            pair_id="reasoning_0002",
        ),
    ]

    content = [
        make_content_record(
            pair_id="reasoning_0001",
        ),
        make_content_record(
            pair_id="reasoning_0002",
            source_question="What is 3 + 3?",
            source_reference_answer="6",
            target_question="3 + 3 neçə edir?",
            target_reference_answer="6",
        ),
    ]

    candidates = generate_final_candidates(
        plan,
        content,
    )

    assert len(candidates) == 4

    assert {
        record.item_id
        for record in candidates
    } == {
        "reasoning_0001_en",
        "reasoning_0001_az",
        "reasoning_0002_en",
        "reasoning_0002_az",
    }


def test_validate_generated_candidate_ids() -> None:
    plan = [
        make_plan_record(),
    ]

    content = [
        make_content_record(),
    ]

    candidates = generate_final_candidates(
        plan,
        content,
    )

    validate_generated_candidate_ids(
        candidates
    )


def test_summarize_final_candidates() -> None:
    plan = [
        make_plan_record(
            pair_id="reasoning_0001",
        ),
        make_plan_record(
            pair_id="unanswerable_0001",
            task="unanswerable",
            category="missing_information",
            is_answerable=False,
        ),
    ]

    content = [
        make_content_record(
            pair_id="reasoning_0001",
        ),
        make_content_record(
            pair_id="unanswerable_0001",
            source_question="How fast did the car arrive?",
            source_reference_answer="Cannot be determined.",
            target_question="Avtomobil nə qədər sürətlə çatdı?",
            target_reference_answer="Müəyyən etmək mümkün deyil.",
        ),
    ]

    candidates = generate_final_candidates(
        plan,
        content,
    )

    summary = summarize_final_candidates(
        candidates
    )

    assert summary == {
        "total_records": 4,
        "total_pairs": 2,
        "by_language": {
            "en": 2,
            "az": 2,
        },
        "by_task": {
            "reasoning": 2,
            "unanswerable": 2,
        },
        "by_review_status": {
            "pending": 4,
        },
    }


def test_save_final_candidates(
    tmp_path: Path,
) -> None:
    plan = [
        make_plan_record(),
    ]

    content = [
        make_content_record(),
    ]

    candidates = generate_final_candidates(
        plan,
        content,
    )

    output_path = (
        tmp_path
        / "final_candidates.jsonl"
    )

    save_final_candidates(
        candidates,
        output_path,
    )

    assert output_path.exists()

    lines = output_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert len(lines) == 2

    first = json.loads(
        lines[0]
    )

    assert first["item_id"] == "reasoning_0001_en"


def test_save_final_candidates_rejects_existing_file(
    tmp_path: Path,
) -> None:
    plan = [
        make_plan_record(),
    ]

    content = [
        make_content_record(),
    ]

    candidates = generate_final_candidates(
        plan,
        content,
    )

    output_path = (
        tmp_path
        / "final_candidates.jsonl"
    )

    save_final_candidates(
        candidates,
        output_path,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        save_final_candidates(
            candidates,
            output_path,
        )


def test_load_generation_plan(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "plan.jsonl"
    )

    path.write_text(
        json.dumps(
            {
                "pair_id": "reasoning_0001",
                "task": "reasoning",
                "category": "logical_reasoning",
                "difficulty": "medium",
                "source_language": "en",
                "target_language": "az",
                "is_answerable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_generation_plan(
        path
    )

    assert len(records) == 1
    assert records[0].pair_id == "reasoning_0001"


def test_load_candidate_content(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "content.jsonl"
    )

    path.write_text(
        json.dumps(
            {
                "pair_id": "reasoning_0001",
                "source_question": "What is 2 + 2?",
                "source_reference_answer": "4",
                "target_question": "2 + 2 neçə edir?",
                "target_reference_answer": "4",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_candidate_content(
        path
    )

    assert len(records) == 1
    assert records[0].pair_id == "reasoning_0001"
    assert records[0].target_question == "2 + 2 neçə edir?"


def test_load_candidate_content_rejects_empty_field(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "content.jsonl"
    )

    path.write_text(
        json.dumps(
            {
                "pair_id": "reasoning_0001",
                "source_question": "",
                "source_reference_answer": "4",
                "target_question": "2 + 2 neçə edir?",
                "target_reference_answer": "4",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        load_candidate_content(
            path
        )
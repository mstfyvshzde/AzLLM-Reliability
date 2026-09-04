"""Benchmark generation plan modülünün testleri."""

from pathlib import Path

import pytest

from src.data.generate_benchmark_plan import (
    BenchmarkPlanRecord,
    allocate_counts,
    build_difficulty_pool,
    build_task_category_slots,
    generate_benchmark_plan,
    save_benchmark_plan,
    summarize_benchmark_plan,
)


def make_config() -> dict:
    """Testlerde kullanılacak küçük benchmark config'i üretir."""

    return {
        "benchmark": {
            "name": "test_benchmark",
            "version": "1.0",
            "languages": {
                "source": "en",
                "target": "az",
            },
            "total_pairs": 10,
            "total_records": 20,
            "seed": 17,
        },
        "pairing": {
            "require_complete_pairs": True,
            "require_same_task": True,
            "require_semantic_equivalence": True,
            "pair_id_format": "{task}_{index:04d}",
            "item_id_format": "{pair_id}_{language}",
        },
        "tasks": {
            "reasoning": {
                "pairs": 5,
                "categories": {
                    "comparative_reasoning": 3,
                    "logical_reasoning": 2,
                },
            },
            "unanswerable": {
                "pairs": 5,
                "categories": {
                    "missing_information": 3,
                    "false_premise": 2,
                },
            },
        },
        "difficulty": {
            "easy": 0.3,
            "medium": 0.4,
            "hard": 0.3,
        },
        "answerability": {
            "reasoning": True,
            "unanswerable": False,
        },
        "review": {
            "required_status": "approved",
            "require_pair_review": True,
            "require_semantic_equivalence_review": True,
            "require_language_quality_review": True,
            "reject_if": [],
        },
        "splits": {
            "strategy": "pair_aware",
            "train": 0.7,
            "dev": 0.15,
            "test": 0.15,
            "preserve_task_distribution": True,
            "preserve_category_distribution": True,
            "preserve_difficulty_distribution": True,
            "seed": 17,
        },
    }


def test_allocate_counts_exact_distribution() -> None:
    counts = allocate_counts(
        10,
        {
            "easy": 0.3,
            "medium": 0.4,
            "hard": 0.3,
        },
    )

    assert counts == {
        "easy": 3,
        "medium": 4,
        "hard": 3,
    }


def test_allocate_counts_preserves_total() -> None:
    counts = allocate_counts(
        7,
        {
            "a": 0.5,
            "b": 0.3,
            "c": 0.2,
        },
    )

    assert sum(counts.values()) == 7


def test_allocate_counts_rejects_negative_total() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        allocate_counts(
            -1,
            {
                "a": 1.0,
            },
        )


def test_allocate_counts_rejects_empty_distribution() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        allocate_counts(
            10,
            {},
        )


def test_allocate_counts_rejects_invalid_distribution_sum() -> None:
    with pytest.raises(
        ValueError,
        match="must sum to 1.0",
    ):
        allocate_counts(
            10,
            {
                "a": 0.7,
                "b": 0.2,
            },
        )


def test_build_difficulty_pool() -> None:
    pool = build_difficulty_pool(
        10,
        {
            "easy": 0.3,
            "medium": 0.4,
            "hard": 0.3,
        },
    )

    assert len(pool) == 10
    assert pool.count("easy") == 3
    assert pool.count("medium") == 4
    assert pool.count("hard") == 3


def test_build_task_category_slots() -> None:
    config = make_config()

    slots = build_task_category_slots(
        config
    )

    assert len(slots) == 10

    assert slots.count(
        (
            "reasoning",
            "comparative_reasoning",
        )
    ) == 3

    assert slots.count(
        (
            "reasoning",
            "logical_reasoning",
        )
    ) == 2

    assert slots.count(
        (
            "unanswerable",
            "missing_information",
        )
    ) == 3

    assert slots.count(
        (
            "unanswerable",
            "false_premise",
        )
    ) == 2


def test_generate_benchmark_plan_size() -> None:
    config = make_config()

    plan = generate_benchmark_plan(
        config
    )

    assert len(plan) == 10


def test_generate_benchmark_plan_languages() -> None:
    plan = generate_benchmark_plan(
        make_config()
    )

    assert all(
        record.source_language == "en"
        for record in plan
    )

    assert all(
        record.target_language == "az"
        for record in plan
    )


def test_generate_benchmark_plan_answerability() -> None:
    plan = generate_benchmark_plan(
        make_config()
    )

    reasoning_records = [
        record
        for record in plan
        if record.task == "reasoning"
    ]

    unanswerable_records = [
        record
        for record in plan
        if record.task == "unanswerable"
    ]

    assert all(
        record.is_answerable
        for record in reasoning_records
    )

    assert all(
        not record.is_answerable
        for record in unanswerable_records
    )


def test_generate_benchmark_plan_pair_ids_are_unique() -> None:
    plan = generate_benchmark_plan(
        make_config()
    )

    pair_ids = [
        record.pair_id
        for record in plan
    ]

    assert len(pair_ids) == len(set(pair_ids))


def test_generate_benchmark_plan_pair_id_format() -> None:
    plan = generate_benchmark_plan(
        make_config()
    )

    reasoning_ids = [
        record.pair_id
        for record in plan
        if record.task == "reasoning"
    ]

    assert reasoning_ids == [
        "reasoning_0001",
        "reasoning_0002",
        "reasoning_0003",
        "reasoning_0004",
        "reasoning_0005",
    ]


def test_generate_benchmark_plan_is_deterministic() -> None:
    config = make_config()

    first = generate_benchmark_plan(
        config
    )

    second = generate_benchmark_plan(
        config
    )

    assert first == second


def test_generate_benchmark_plan_preserves_difficulty_distribution() -> None:
    plan = generate_benchmark_plan(
        make_config()
    )

    difficulties = [
        record.difficulty
        for record in plan
    ]

    assert difficulties.count("easy") == 3
    assert difficulties.count("medium") == 4
    assert difficulties.count("hard") == 3


def test_summarize_benchmark_plan() -> None:
    plan = generate_benchmark_plan(
        make_config()
    )

    summary = summarize_benchmark_plan(
        plan
    )

    assert summary["total_pairs"] == 10

    assert summary["by_task"] == {
        "reasoning": 5,
        "unanswerable": 5,
    }

    assert summary["by_difficulty"] == {
        "easy": 3,
        "medium": 4,
        "hard": 3,
    }

    assert summary["answerable"] == 5
    assert summary["unanswerable"] == 5


def test_benchmark_plan_record_to_dict() -> None:
    record = BenchmarkPlanRecord(
        pair_id="reasoning_0001",
        task="reasoning",
        category="logical_reasoning",
        difficulty="medium",
        source_language="en",
        target_language="az",
        is_answerable=True,
    )

    assert record.to_dict() == {
        "pair_id": "reasoning_0001",
        "task": "reasoning",
        "category": "logical_reasoning",
        "difficulty": "medium",
        "source_language": "en",
        "target_language": "az",
        "is_answerable": True,
    }


def test_save_benchmark_plan(
    tmp_path: Path,
) -> None:
    plan = generate_benchmark_plan(
        make_config()
    )

    output_path = (
        tmp_path
        / "benchmark_plan.jsonl"
    )

    save_benchmark_plan(
        plan,
        output_path,
    )

    assert output_path.exists()

    lines = output_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert len(lines) == 10


def test_save_benchmark_plan_rejects_existing_file(
    tmp_path: Path,
) -> None:
    plan = generate_benchmark_plan(
        make_config()
    )

    output_path = (
        tmp_path
        / "benchmark_plan.jsonl"
    )

    save_benchmark_plan(
        plan,
        output_path,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        save_benchmark_plan(
            plan,
            output_path,
        )


def test_save_benchmark_plan_allows_overwrite(
    tmp_path: Path,
) -> None:
    plan = generate_benchmark_plan(
        make_config()
    )

    output_path = (
        tmp_path
        / "benchmark_plan.jsonl"
    )

    save_benchmark_plan(
        plan,
        output_path,
    )

    save_benchmark_plan(
        plan,
        output_path,
        overwrite=True,
    )

    assert output_path.exists()
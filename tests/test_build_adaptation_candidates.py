from pathlib import Path

import pytest

from src.data.build_adaptation_candidates import (
    allocate_category_counts,
    build_candidate_records,
    load_adaptation_mix,
)


def test_allocate_category_counts() -> None:
    counts = allocate_category_counts(
        total_records=100,
        data_mix={
            "instruction_following": 0.25,
            "semantic_understanding": 0.25,
            "reasoning": 0.20,
            "unanswerable_abstention": 0.20,
            "factual_knowledge": 0.10,
        },
    )

    assert counts == {
        "instruction_following": 25,
        "semantic_understanding": 25,
        "reasoning": 20,
        "unanswerable_abstention": 20,
        "factual_knowledge": 10,
    }


def test_allocate_category_counts_preserves_total() -> None:
    counts = allocate_category_counts(
        total_records=7,
        data_mix={
            "a": 0.5,
            "b": 0.5,
        },
    )

    assert sum(
        counts.values()
    ) == 7


def test_allocate_category_counts_rejects_invalid_total() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        allocate_category_counts(
            total_records=0,
            data_mix={
                "a": 1.0,
            },
        )


def test_allocate_category_counts_rejects_empty_mix() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        allocate_category_counts(
            total_records=10,
            data_mix={},
        )


def test_allocate_category_counts_rejects_invalid_mix_sum() -> None:
    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        allocate_category_counts(
            total_records=10,
            data_mix={
                "a": 0.4,
                "b": 0.4,
            },
        )


def test_build_candidate_records() -> None:
    records = build_candidate_records(
        total_records=4,
        data_mix={
            "instruction_following": 0.5,
            "reasoning": 0.5,
        },
    )

    assert len(records) == 4

    assert records[0].item_id == "adapt_00001"

    assert all(
        record.language == "az"
        for record in records
    )

    assert all(
        record.source == "synthetic"
        for record in records
    )

    assert all(
        record.metadata["review_status"]
        == "pending"
        for record in records
    )

    assert all(
        record.metadata["generation_status"]
        == "pending"
        for record in records
    )


def test_load_adaptation_mix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mix.yaml"

    path.write_text(
        "data_mix:\n"
        "  instruction_following: 1.0\n",
        encoding="utf-8",
    )

    config = load_adaptation_mix(
        path
    )

    assert (
        config["data_mix"]["instruction_following"]
        == 1.0
    )
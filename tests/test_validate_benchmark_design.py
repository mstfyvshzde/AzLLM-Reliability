"""Final benchmark design validation testleri."""

from copy import deepcopy

import pytest

from src.data.validate_benchmark_design import (
    validate_answerability,
    validate_benchmark_design,
    validate_category_totals,
    validate_difficulty_distribution,
    validate_record_total,
    validate_split_distribution,
    validate_task_pair_total,
)


def make_valid_config() -> dict:
    """Testlerde kullanılacak geçerli benchmark config'i üretir."""

    return {
        "benchmark": {
            "name": "azllm_reliability_final",
            "version": "1.0",
            "languages": {
                "source": "en",
                "target": "az",
            },
            "total_pairs": 10,
            "total_records": 20,
            "seed": 17,
        },
        "tasks": {
            "reasoning": {
                "pairs": 5,
                "categories": {
                    "logical_reasoning": 3,
                    "comparative_reasoning": 2,
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
        "splits": {
            "strategy": "pair_aware",
            "train": 0.7,
            "dev": 0.15,
            "test": 0.15,
        },
    }


def test_validate_task_pair_total() -> None:
    config = make_valid_config()

    validate_task_pair_total(config)


def test_validate_task_pair_total_rejects_mismatch() -> None:
    config = make_valid_config()
    config["benchmark"]["total_pairs"] = 11

    with pytest.raises(
        ValueError,
        match="Task pair total does not match",
    ):
        validate_task_pair_total(config)


def test_validate_category_totals() -> None:
    config = make_valid_config()

    validate_category_totals(config)


def test_validate_category_totals_rejects_mismatch() -> None:
    config = make_valid_config()
    config["tasks"]["reasoning"]["categories"][
        "logical_reasoning"
    ] = 4

    with pytest.raises(
        ValueError,
        match="Category total mismatch",
    ):
        validate_category_totals(config)


def test_validate_difficulty_distribution() -> None:
    config = make_valid_config()

    validate_difficulty_distribution(config)


def test_validate_difficulty_distribution_rejects_invalid_sum() -> None:
    config = make_valid_config()
    config["difficulty"]["hard"] = 0.4

    with pytest.raises(
        ValueError,
        match="Difficulty distribution must sum to 1.0",
    ):
        validate_difficulty_distribution(config)


def test_validate_record_total() -> None:
    config = make_valid_config()

    validate_record_total(config)


def test_validate_record_total_rejects_mismatch() -> None:
    config = make_valid_config()
    config["benchmark"]["total_records"] = 19

    with pytest.raises(
        ValueError,
        match="Benchmark total_records mismatch",
    ):
        validate_record_total(config)


def test_validate_split_distribution() -> None:
    config = make_valid_config()

    validate_split_distribution(config)


def test_validate_split_distribution_rejects_invalid_sum() -> None:
    config = make_valid_config()
    config["splits"]["test"] = 0.2

    with pytest.raises(
        ValueError,
        match="Split distribution must sum to 1.0",
    ):
        validate_split_distribution(config)


def test_validate_answerability() -> None:
    config = make_valid_config()

    validate_answerability(config)


def test_validate_answerability_rejects_missing_task() -> None:
    config = make_valid_config()
    del config["answerability"]["reasoning"]

    with pytest.raises(
        ValueError,
        match="Missing answerability definitions",
    ):
        validate_answerability(config)


def test_validate_answerability_rejects_unexpected_task() -> None:
    config = make_valid_config()
    config["answerability"]["unknown_task"] = True

    with pytest.raises(
        ValueError,
        match="Unexpected answerability definitions",
    ):
        validate_answerability(config)


def test_validate_answerability_rejects_non_boolean() -> None:
    config = make_valid_config()
    config["answerability"]["reasoning"] = "yes"

    with pytest.raises(
        ValueError,
        match="must be boolean",
    ):
        validate_answerability(config)


def test_validate_benchmark_design() -> None:
    config = make_valid_config()

    validate_benchmark_design(config)


def test_validate_benchmark_design_rejects_invalid_config() -> None:
    config = deepcopy(
        make_valid_config()
    )

    config["benchmark"]["total_pairs"] = 99

    with pytest.raises(ValueError):
        validate_benchmark_design(config)
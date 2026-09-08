from pathlib import Path
import random

import pytest

from src.data.generate_adaptation_data import (
    choose_subcategory,
    generate_example,
    generate_records,
    load_candidates,
    load_yaml,
)


def test_choose_subcategory_cycles_deterministically() -> None:
    config = {
        "subcategories": [
            "a",
            "b",
            "c",
        ]
    }

    assert choose_subcategory(
        category="test",
        category_config=config,
        index=0,
    ) == "a"

    assert choose_subcategory(
        category="test",
        category_config=config,
        index=1,
    ) == "b"

    assert choose_subcategory(
        category="test",
        category_config=config,
        index=3,
    ) == "a"


def test_choose_subcategory_rejects_empty_list() -> None:
    with pytest.raises(
        ValueError,
        match="Empty subcategory list",
    ):
        choose_subcategory(
            category="test",
            category_config={
                "subcategories": [],
            },
            index=0,
        )


def test_generate_example_instruction_following() -> None:
    rng = random.Random(17)

    instruction, response = generate_example(
        category="instruction_following",
        subcategory="format_following",
        generation_config={
            "categories": {},
        },
        rng=rng,
        variant_index=0,
    )

    assert instruction
    assert response
    assert response.startswith("(")
    assert response.endswith(")")


def test_generate_example_reasoning() -> None:
    rng = random.Random(17)

    instruction, response = generate_example(
        category="reasoning",
        subcategory="logical_reasoning",
        generation_config={
            "categories": {},
        },
        rng=rng,
        variant_index=0,
    )

    assert "lalələr" in instruction
    assert response == "Bəli."


def test_generate_example_unanswerable() -> None:
    rng = random.Random(17)

    config = {
        "categories": {
            "unanswerable_abstention": {
                "canonical_response": (
                    "Verilən məlumatlardan müəyyən etmək mümkün deyil."
                )
            }
        }
    }

    instruction, response = generate_example(
        category="unanswerable_abstention",
        subcategory="missing_information",
        generation_config=config,
        rng=rng,
        variant_index=0,
    )

    assert instruction
    assert (
        response
        == "Verilən məlumatlardan müəyyən etmək mümkün deyil."
    )


def test_generate_records() -> None:
    candidates = [
        {
            "item_id": "adapt_00001",
            "language": "az",
            "category": "instruction_following",
            "instruction": "",
            "response": "",
            "source": "synthetic",
            "metadata": {
                "review_status": "pending",
                "generation_status": "pending",
            },
        },
        {
            "item_id": "adapt_00002",
            "language": "az",
            "category": "reasoning",
            "instruction": "",
            "response": "",
            "source": "synthetic",
            "metadata": {
                "review_status": "pending",
                "generation_status": "pending",
            },
        },
    ]

    config = {
        "generation": {
            "seed": 17,
        },
        "categories": {
            "instruction_following": {
                "subcategories": [
                    "format_following",
                ]
            },
            "reasoning": {
                "subcategories": [
                    "logical_reasoning",
                ]
            },
        },
    }

    records = generate_records(
        candidates=candidates,
        generation_config=config,
    )

    assert len(records) == 2

    assert records[0].item_id == "adapt_00001"
    assert records[1].item_id == "adapt_00002"

    assert records[0].instruction
    assert records[0].response

    assert records[1].instruction
    assert records[1].response

    assert (
        records[0].metadata["generation_status"]
        == "generated"
    )

    assert (
        records[1].metadata["generation_status"]
        == "generated"
    )

    assert (
        records[0].metadata["review_status"]
        == "pending"
    )


def test_generate_records_is_deterministic() -> None:
    candidates = [
        {
            "item_id": "adapt_00001",
            "language": "az",
            "category": "factual_knowledge",
            "instruction": "",
            "response": "",
            "source": "synthetic",
            "metadata": {},
        }
    ]

    config = {
        "generation": {
            "seed": 17,
        },
        "categories": {
            "factual_knowledge": {
                "subcategories": [
                    "quantitative_knowledge",
                ]
            },
        },
    }

    first = generate_records(
        candidates=candidates,
        generation_config=config,
    )

    second = generate_records(
        candidates=candidates,
        generation_config=config,
    )

    assert (
        first[0].instruction
        == second[0].instruction
    )

    assert (
        first[0].response
        == second[0].response
    )


def test_load_yaml(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"

    path.write_text(
        "generation:\n"
        "  seed: 17\n",
        encoding="utf-8",
    )

    config = load_yaml(
        path
    )

    assert config["generation"]["seed"] == 17


def test_load_candidates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidates.jsonl"

    path.write_text(
        '{"item_id":"adapt_00001","language":"az",'
        '"category":"reasoning","instruction":"",'
        '"response":"","source":"synthetic","metadata":{}}\n',
        encoding="utf-8",
    )

    rows = load_candidates(
        path
    )

    assert len(rows) == 1
    assert rows[0]["item_id"] == "adapt_00001"
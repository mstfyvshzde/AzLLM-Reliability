from pathlib import Path

import pytest

from src.data.validate_adaptation_data import (
    count_words,
    load_config,
    load_jsonl,
    validate_record,
    validate_records,
    validate_unique_instructions,
    validate_unique_item_ids,
    validate_unique_pairs,
)


def make_config() -> dict:
    return {
        "language": "az",
        "generation": {
            "require_non_empty_instruction": True,
            "require_non_empty_response": True,
            "max_instruction_words": 120,
            "max_response_words": 180,
        },
        "categories": {
            "reasoning": {
                "subcategories": [
                    "logical_reasoning",
                    "arithmetic_reasoning",
                ]
            }
        },
        "quality": {
            "require_unique_instruction": True,
            "require_unique_response_pair": True,
        },
    }


def make_row(
    item_id: str = "adapt_00001",
    instruction: str = "Bütün A-lar B-dir. Nəticə doğrudurmu?",
    response: str = "Bəli.",
) -> dict:
    return {
        "item_id": item_id,
        "language": "az",
        "category": "reasoning",
        "instruction": instruction,
        "response": response,
        "source": "synthetic",
        "metadata": {
            "review_status": "pending",
            "generation_status": "generated",
            "subcategory": "logical_reasoning",
        },
    }


def test_count_words() -> None:
    assert count_words(
        "bir iki üç dörd"
    ) == 4


def test_validate_record_passes() -> None:
    validate_record(
        row=make_row(),
        config=make_config(),
    )


def test_validate_record_rejects_empty_instruction() -> None:
    row = make_row(
        instruction="",
    )

    with pytest.raises(
        ValueError,
        match="Empty instruction",
    ):
        validate_record(
            row=row,
            config=make_config(),
        )


def test_validate_record_rejects_wrong_language() -> None:
    row = make_row()
    row["language"] = "en"

    with pytest.raises(
        ValueError,
        match="Invalid language",
    ):
        validate_record(
            row=row,
            config=make_config(),
        )


def test_validate_record_rejects_invalid_subcategory() -> None:
    row = make_row()
    row["metadata"]["subcategory"] = "unknown"

    with pytest.raises(
        ValueError,
        match="Invalid subcategory",
    ):
        validate_record(
            row=row,
            config=make_config(),
        )


def test_validate_unique_item_ids_rejects_duplicate() -> None:
    rows = [
        make_row(
            item_id="adapt_00001",
        ),
        make_row(
            item_id="adapt_00001",
            instruction="Başqa sual",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate item_id",
    ):
        validate_unique_item_ids(
            rows
        )


def test_validate_unique_instructions_rejects_duplicate() -> None:
    rows = [
        make_row(
            item_id="adapt_00001",
            instruction="Sadə sual",
        ),
        make_row(
            item_id="adapt_00002",
            instruction="  SADƏ   SUAL ",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate instruction",
    ):
        validate_unique_instructions(
            rows
        )


def test_validate_unique_pairs_rejects_duplicate() -> None:
    rows = [
        make_row(
            item_id="adapt_00001",
            instruction="Sual",
            response="Cavab",
        ),
        make_row(
            item_id="adapt_00002",
            instruction=" sual ",
            response=" cavab ",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate instruction-response pair",
    ):
        validate_unique_pairs(
            rows
        )


def test_validate_records_passes() -> None:
    rows = [
        make_row(
            item_id="adapt_00001",
            instruction="Birinci sual",
            response="Bəli.",
        ),
        make_row(
            item_id="adapt_00002",
            instruction="İkinci sual",
            response="Xeyr.",
        ),
    ]

    validate_records(
        rows=rows,
        config=make_config(),
    )


def test_load_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data.jsonl"

    path.write_text(
        '{"item_id":"adapt_00001"}\n',
        encoding="utf-8",
    )

    rows = load_jsonl(
        path
    )

    assert len(rows) == 1
    assert rows[0]["item_id"] == "adapt_00001"


def test_load_config(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"

    path.write_text(
        "language: az\n",
        encoding="utf-8",
    )

    config = load_config(
        path
    )

    assert config["language"] == "az"
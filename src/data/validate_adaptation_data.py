"""Adaptation datasını training öncesi doğrular.

Bu modül generated adaptation kayıtlarında schema,
uniqueness, boş alan, kategori, dil ve uzunluk
kontrollerini uygular.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """JSONL kayıtlarını yükler."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}."
                ) from error

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} must be a JSON object."
                )

            rows.append(row)

    return rows


def load_config(
    path: Path,
) -> dict[str, Any]:
    """Generation config dosyasını yükler."""

    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Config must be a mapping."
        )

    return config


def count_words(
    text: str,
) -> int:
    """Metindeki whitespace-separated kelime sayısını döndürür."""

    return len(
        text.split()
    )


def validate_required_fields(
    row: dict[str, Any],
) -> None:
    """Zorunlu alanların varlığını kontrol eder."""

    required_fields = {
        "item_id",
        "language",
        "category",
        "instruction",
        "response",
        "source",
        "metadata",
    }

    missing = required_fields - row.keys()

    if missing:
        raise ValueError(
            f"Missing required fields for "
            f"{row.get('item_id', '<unknown>')}: "
            f"{sorted(missing)}"
        )


def validate_record(
    row: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Tek bir adaptation kaydını doğrular."""

    validate_required_fields(
        row
    )

    item_id = row["item_id"]

    if row["language"] != config.get(
        "language"
    ):
        raise ValueError(
            f"Invalid language for {item_id}: "
            f"{row['language']}"
        )

    categories = config.get(
        "categories"
    )

    if not isinstance(
        categories,
        dict,
    ):
        raise ValueError(
            "Config must contain categories."
        )

    if row["category"] not in categories:
        raise ValueError(
            f"Unknown category for {item_id}: "
            f"{row['category']}"
        )

    instruction = row["instruction"]
    response = row["response"]

    if not isinstance(
        instruction,
        str,
    ):
        raise ValueError(
            f"Instruction must be a string: {item_id}"
        )

    if not isinstance(
        response,
        str,
    ):
        raise ValueError(
            f"Response must be a string: {item_id}"
        )

    generation = config.get(
        "generation",
        {},
    )

    if generation.get(
        "require_non_empty_instruction",
        True,
    ):
        if not instruction.strip():
            raise ValueError(
                f"Empty instruction: {item_id}"
            )

    if generation.get(
        "require_non_empty_response",
        True,
    ):
        if not response.strip():
            raise ValueError(
                f"Empty response: {item_id}"
            )

    max_instruction_words = generation.get(
        "max_instruction_words"
    )

    if (
        isinstance(max_instruction_words, int)
        and count_words(instruction)
        > max_instruction_words
    ):
        raise ValueError(
            f"Instruction too long: {item_id}"
        )

    max_response_words = generation.get(
        "max_response_words"
    )

    if (
        isinstance(max_response_words, int)
        and count_words(response)
        > max_response_words
    ):
        raise ValueError(
            f"Response too long: {item_id}"
        )

    metadata = row["metadata"]

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            f"Metadata must be a mapping: {item_id}"
        )

    if (
        metadata.get("generation_status")
        != "generated"
    ):
        raise ValueError(
            f"Invalid generation_status: {item_id}"
        )

    if (
        metadata.get("review_status")
        != "pending"
    ):
        raise ValueError(
            f"Invalid review_status: {item_id}"
        )

    subcategory = metadata.get(
        "subcategory"
    )

    expected_subcategories = categories[
        row["category"]
    ].get(
        "subcategories",
        [],
    )

    if subcategory not in expected_subcategories:
        raise ValueError(
            f"Invalid subcategory for {item_id}: "
            f"{subcategory}"
        )


def validate_unique_item_ids(
    rows: list[dict[str, Any]],
) -> None:
    """Duplicate item_id bulunmadığını doğrular."""

    seen: set[str] = set()

    for row in rows:
        item_id = row["item_id"]

        if item_id in seen:
            raise ValueError(
                f"Duplicate item_id: {item_id}"
            )

        seen.add(
            item_id
        )


def validate_unique_instructions(
    rows: list[dict[str, Any]],
) -> None:
    """Duplicate instruction bulunmadığını doğrular."""

    seen: dict[str, str] = {}

    for row in rows:
        normalized = " ".join(
            row["instruction"]
            .strip()
            .lower()
            .split()
        )

        if normalized in seen:
            raise ValueError(
                "Duplicate instruction: "
                f"{seen[normalized]} and "
                f"{row['item_id']}"
            )

        seen[normalized] = row["item_id"]


def validate_unique_pairs(
    rows: list[dict[str, Any]],
) -> None:
    """Duplicate instruction-response pair bulunmadığını doğrular."""

    seen: dict[
        tuple[str, str],
        str,
    ] = {}

    for row in rows:
        instruction = " ".join(
            row["instruction"]
            .strip()
            .lower()
            .split()
        )

        response = " ".join(
            row["response"]
            .strip()
            .lower()
            .split()
        )

        key = (
            instruction,
            response,
        )

        if key in seen:
            raise ValueError(
                "Duplicate instruction-response pair: "
                f"{seen[key]} and "
                f"{row['item_id']}"
            )

        seen[key] = row["item_id"]


def validate_records(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Bütün adaptation kayıtlarını doğrular."""

    if not rows:
        raise ValueError(
            "Adaptation dataset cannot be empty."
        )

    for row in rows:
        validate_record(
            row=row,
            config=config,
        )

    validate_unique_item_ids(
        rows
    )

    quality = config.get(
        "quality",
        {},
    )

    if quality.get(
        "require_unique_instruction",
        True,
    ):
        validate_unique_instructions(
            rows
        )

    if quality.get(
        "require_unique_response_pair",
        True,
    ):
        validate_unique_pairs(
            rows
        )


def parse_arguments() -> argparse.Namespace:
    """CLI argumentlerini parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate Azerbaijani adaptation data."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/adaptation/generation_v1.0.yaml"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_arguments()

    rows = load_jsonl(
        args.input
    )

    config = load_config(
        args.config
    )

    validate_records(
        rows=rows,
        config=config,
    )

    print(
        f"✅ Adaptation validation passed: "
        f"{len(rows)} records."
    )


if __name__ == "__main__":
    main()
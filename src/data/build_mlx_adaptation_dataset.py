"""Adaptation kayıtlarını MLX-LM chat dataset formatına dönüştürür.

Pilot adaptation datasını deterministic şekilde train/valid olarak böler.

Output:

    train.jsonl
    valid.jsonl

Her kayıt MLX-LM için chat messages formatındadır.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_records(
    path: Path,
) -> list[dict[str, Any]]:
    """JSONL adaptation kayıtlarını yükler."""

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            records.append(
                json.loads(line)
            )

    return records


def to_mlx_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Adaptation kaydını MLX chat formatına dönüştürür."""

    return {
        "messages": [
            {
                "role": "user",
                "content": record["instruction"],
            },
            {
                "role": "assistant",
                "content": record["response"],
            },
        ],
        "metadata": {
            "item_id": record["item_id"],
            "category": record["category"],
            "subcategory": record[
                "metadata"
            ]["subcategory"],
        },
    }


def stratified_split(
    records: list[dict[str, Any]],
    valid_ratio: float,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Kayıtları category bazında train/valid olarak böler."""

    rng = random.Random(
        seed
    )

    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    for record in records:
        groups[
            record["category"]
        ].append(
            record
        )

    train_records: list[
        dict[str, Any]
    ] = []

    valid_records: list[
        dict[str, Any]
    ] = []

    for category in sorted(
        groups
    ):
        group = list(
            groups[category]
        )

        rng.shuffle(
            group
        )

        valid_count = round(
            len(group)
            * valid_ratio
        )

        valid_count = max(
            1,
            valid_count,
        )

        valid_records.extend(
            group[:valid_count]
        )

        train_records.extend(
            group[valid_count:]
        )

    rng.shuffle(
        train_records
    )

    rng.shuffle(
        valid_records
    )

    return (
        train_records,
        valid_records,
    )


def save_jsonl(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    """Kayıtları JSONL dosyasına yazar."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_dataset(
    input_path: Path,
    output_dir: Path,
    valid_ratio: float,
    seed: int,
) -> None:
    """MLX pilot train/valid datasetini oluşturur."""

    records = load_records(
        input_path
    )

    train_records, valid_records = (
        stratified_split(
            records=records,
            valid_ratio=valid_ratio,
            seed=seed,
        )
    )

    train_mlx = [
        to_mlx_record(
            record
        )
        for record in train_records
    ]

    valid_mlx = [
        to_mlx_record(
            record
        )
        for record in valid_records
    ]

    save_jsonl(
        train_mlx,
        output_dir / "train.jsonl",
    )

    save_jsonl(
        valid_mlx,
        output_dir / "valid.jsonl",
    )

    print(
        f"Total: {len(records)}"
    )

    print(
        f"Train: {len(train_mlx)}"
    )

    print(
        f"Valid: {len(valid_mlx)}"
    )


def parse_argumens() -> argparse.Namespace:
    """CLI argümanlarını parse eder."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_argumens()

    build_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
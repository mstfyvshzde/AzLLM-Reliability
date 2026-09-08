"""Adaptation candidate kayıtlarını oluşturur.

Bu modül adaptation data mix config dosyasını okuyarak
kategori bazlı hedef sayıları hesaplar ve pending review
durumunda sentetik candidate kayıtları üretir.

Bu aşama yalnızca candidate iskeleti oluşturur.
Gerçek instruction/response içerikleri sonraki generation
aşamasında doldurulur.
"""


from __future__ import annotations

import argparse 
import json
from pathlib import Path
from typing import Any

import yaml

from src.data.adaptation_record import AdaptationRecord


def load_adaptation_mix(
    config_path: Path
) -> dict[str, Any]:  
    """Adaptation mix YAML config dosyasını yükler."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Adaptation mix config not found: {config_path}"
        )

    with config_path.open('r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Adaptation mix config must be a mapping."
        )

    return data


def allocate_category_counts(
    total_records: int,
    data_mix: dict[str, float]
) -> dict[str, int]:
    """Toplam kayıt sayısını kategori oranlarına dağıtır."""

    if total_records <= 0:
        raise ValueError(
            "total_records must be greater than zero."
        )

    if not data_mix:
        raise ValueError(
            "data_mix cannot be empty."
        )

    total_weight = sum(
        data_mix.values()
    )

    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(
            "data_mix weights must sum to 1.0."
        )

    raw_counts = {
        category: total_records * weight
        for category, weight in data_mix.items()
    }

    counts = {
        category: int(value)
        for category, value in raw_counts.items()
    }

    remainder = total_records - sum(counts.values())

    ranked = sorted(
        raw_counts,
        key=lambda category: (
            raw_counts[category]
            -counts[category]
        ),
        reverse=True
    )

    for category in ranked[:remainder]:
        counts[category] += 1

    return counts


def build_candidate_records(
    total_records: int,
    data_mix: dict[str, float],
) -> list[AdaptationRecord]:
    """Kategori hedeflerine göre pending candidate kayıtları üretir."""

    counts = allocate_category_counts(
        total_records=total_records,
        data_mix=data_mix,
    )

    records: list[AdaptationRecord] = []

    index = 1

    for category, count in counts.items():
        for _ in range(count):
            record = AdaptationRecord(
                item_id=f"adapt_{index:05d}",
                language="az",
                category=category,
                instruction="",
                response="",
                source="synthetic",
                metadata={
                    "review_status": "pending",
                    "generation_status": "pending",
                },
            )

            records.append(
                record
            )

            index += 1

    return records


def save_records(
    records: list[AdaptationRecord],
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """Adaptation candidate kayıtlarını JSONL olarak kaydeder."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_arguments() -> argparse.Namespace:
    """CLI argumentlerini parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Build Azerbaijani adaptation candidate records."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/adaptation/data_mix_v1.0.yaml"
        ),
    )

    parser.add_argument(
        "--total-records",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Adaptation candidate builder CLI entry point."""

    args = parse_arguments()

    config = load_adaptation_mix(
        args.config
    )

    data_mix = config.get(
        "data_mix"
    )

    if not isinstance(
        data_mix,
        dict,
    ):
        raise ValueError(
            "Config must contain a data_mix mapping."
        )

    records = build_candidate_records(
        total_records=args.total_records,
        data_mix=data_mix,
    )

    save_records(
        records=records,
        output_path=args.output,
        overwrite=args.overwrite,
    )

    print(
        f"Created {len(records)} adaptation candidates."
    )


if __name__ == "__main__":
    main()
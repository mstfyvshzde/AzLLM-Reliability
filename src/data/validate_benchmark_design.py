"""Final benchmark design config'inin tutarlılığını doğrular.

Bu modül configs/benchmark_final.yaml içindeki benchmark tasarımının
matematiksel ve yapısal olarak kendi içinde tutarlı olup olmadığını kontrol eder.

Kontroller:

    1. Task pair toplamı benchmark.total_pairs ile eşleşmeli.
    2. Her task içindeki category pair toplamı task pair sayısına eşit olmalı.
    3. Difficulty oranlarının toplamı 1.0 olmalı.
    4. total_records == total_pairs * language_count olmalı.
    5. Split oranlarının toplamı 1.0 olmalı.
    6. Her task için answerability değeri tanımlı olmalı.
"""


from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(
    "configs/benchmark_final.yaml"
)


def load_benchmark_design(
    config_path: Path = DEFAULT_CONFIG_PATH
) -> dict[str, Any]:
    """Benchmark design YAML dosyasını yükler."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Benchmark design config not found: {config_path}"
        )

    with config_path.open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Benchmark design config must contain a mapping."
        )

    return config



def validate_task_pair_total(
    config: dict[str, Any]
) -> None:
    """Task pair toplamının benchmark total_pairs ile eşleştiğini doğrular."""

    expected_total = config["benchmark"]["total_pairs"]

    actual_total = sum(
        task_config['pairs']
        for task_config in config['tasks'].values()
    )

    if actual_total != expected_total:
        raise ValueError(
            "Task pair total does not match benchmark total_pairs: "
            f"expected={expected_total}, actual={actual_total}"
        )



def validate_category_totals(
    config: dict[str, Any]
) -> None:
    """Her task için category toplamının task pair quota'sına eşit olduğunu doğrular."""

    for task, task_config in config['tasks'].items():
        expected_total = task_config['pairs']

        categories = task_config.get(
            'categories',
            {}
        )

        actual_total = sum(
            categories.values()
        )

        if actual_total != expected_total:
            raise ValueError(
                f"Category total mismatch for task '{task}': "
                f"expected={expected_total}, actual={actual_total}"
            )


def validate_difficulty_distribution(
    config: dict[str, Any]
) -> None:
    """Difficulty oranlarının toplamının 1.0 olduğunu doğrular."""

    difficulty  = config['difficulty']

    total = sum(
        difficulty.values()
    )

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Difficulty distribution must sum to 1.0: "
            f"actual={total}"
        )



def validate_record_total(
    config: dict[str, Any]
) -> None:
    """total_records değerinin pair ve language sayısıyla uyumlu olduğunu doğrular."""


    total_pairs = config["benchmark"][
        "total_pairs"
    ]

    total_records = config["benchmark"][
        "total_records"
    ]

    languages = config["benchmark"][
        "languages"
    ]

    language_count = len(
        languages
    )

    expected_records = (
        total_pairs
        * language_count
    )

    if total_records != expected_records:
        raise ValueError(
            "Benchmark total_records mismatch: "
            f"expected={expected_records}, actual={total_records}"
        )



def validate_split_distribution(
    config: dict[str, Any]
)-> None:
    """Train/dev/test split oranlarının toplamının 1.0 olduğunu doğrular."""

    splits = config['splits']


    total = (
        splits["train"]
        + splits["dev"]
        + splits["test"]
    )

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Split distribution must sum to 1.0: "
            f"actual={total}"
        )


def validate_answerability(
    config: dict[str, Any],
) -> None:
    """Her task için answerability bilgisinin mevcut olduğunu doğrular."""

    tasks = set(
        config["tasks"]
    )

    answerability = config[
        "answerability"
    ]

    answerability_tasks = set(
        answerability
    )

    missing = (
        tasks
        - answerability_tasks
    )

    unexpected = (
        answerability_tasks
        - tasks
    )

    if missing:
        raise ValueError(
            "Missing answerability definitions: "
            f"{sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            "Unexpected answerability definitions: "
            f"{sorted(unexpected)}"
        )

    for task, value in answerability.items():
        if not isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"Answerability for task '{task}' "
                "must be boolean."
            )


def validate_benchmark_design(
    config: dict[str, Any],
) -> None:
    """Final benchmark design'in tüm validation kontrollerini çalıştırır."""

    validate_task_pair_total(
        config
    )

    validate_category_totals(
        config
    )

    validate_difficulty_distribution(
        config
    )

    validate_record_total(
        config
    )

    validate_split_distribution(
        config
    )

    validate_answerability(
        config
    )


def main() -> None:
    """Benchmark design config'ini yükler ve doğrular."""

    config = load_benchmark_design()

    validate_benchmark_design(
        config
    )

    print(
        "Benchmark design validation passed."
    )


if __name__ == "__main__":
    main()
"""Final benchmark için deterministic generation plan üretir.

Bu modül configs/benchmark_final.yaml içindeki quota bilgilerini kullanarak
her semantic pair için task, category ve difficulty ataması yapar.

Amaç doğrudan soru üretmek değildir.

Önce şu türde bir plan oluşturulur:

    {
        "pair_id": "reasoning_0001",
        "task": "reasoning",
        "category": "logical_reasoning",
        "difficulty": "easy",
        "source_language": "en",
        "target_language": "az",
        "is_answerable": True
    }

Bu plan daha sonra candidate generation aşamasında kullanılır.

Önemli:
    - Plan deterministic olmalıdır.
    - Task quota'ları korunmalıdır.
    - Category quota'ları korunmalıdır.
    - Difficulty dağılımı yaklaşık değil, mümkün olduğunca tam korunmalıdır.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.validate_benchmark_design import (
    load_benchmark_design,
    validate_benchmark_design
)


DEFAULT_CONFIG_PATH = Path(
    "configs/benchmark_final.yaml"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/processed/benchmark_generation_plan.jsonl"
)


@dataclass(frozen=True)
class BenchmarkPlanRecord:
    """Tek bir semantic pair için generation plan kaydını temsil eder."""

    pair_id: str
    task: str
    category: str
    difficulty: str
    source_language: str
    target_language: str
    is_answerable: bool

    def to_dict(self) -> dict[str, Any]:
        """Plan kaydını JSON-serializable dictionary biçimine dönüştürür."""

        return asdict(self)



def allocate_counts(
    total: int,
    distribution: dict[str, float]
) -> dict[str, int]:
    """Oransal dağılımdan toplamı tam koruyan integer count üretir.

    Largest-remainder yaklaşımı kullanılır.

    Örnek:

        total = 10

        distribution = {
            "easy": 0.3,
            "medium": 0.4,
            "hard": 0.3,
        }

    Sonuç:

        {
            "easy": 3,
            "medium": 4,
            "hard": 3,
        }
    """

    if total < 0:
        raise ValueError(
            "Total count cannot be negative."
        )

    if not distribution:
        raise ValueError(
            "Distribution cannot be empty."
        )

    distribution_total = sum(
        distribution.values()
    )

    if abs(distribution_total - 1.0) > 1e-9:
        raise ValueError(
            "Distribution must sum to 1.0."
        )

    raw_counts = {
        key: total * ratio
        for key, ratio in distribution.items()
    }

    counts = {
        key: int(value)
        for key, value in raw_counts.items()
    }

    remaining = (
        total 
        - sum(counts.values())
    )

    remainders = sorted(
        distribution,
        key=lambda key:(
            raw_counts[key]
            -counts[key]
        ),
        reverse=True
    )

    for key in remainders[:remaining]:
        counts[key] += 1

    return counts


def build_difficulty_pool(
    total_pairs: int,
    difficulty_distribution: dict[str, float]
) -> list[str]:
    """Difficulty dağılımından toplam pair sayısı kadar difficulty listesi üretir."""

    counts = allocate_counts(
        total_pairs,
        difficulty_distribution
    )

    pool: list[str] = []

    for difficulty, count in counts.items():
        pool.extend(
            [difficulty] * count
        )

    if len(pool) != total_pairs:
        raise ValueError(
            "Difficulty pool size does not match total_pairs."
        )

    return pool


def build_task_category_slots(
    config: dict[str, Any]
) -> list[tuple[str, str]]:
    """Task/category quota'larından pair slot listesi üretir."""

    slots: list[tuple[str, str]] = []

    for task, task_config in config['tasks'].items():
        categories = task_config['categories']

        for category, count in categories.items():
            slots.extend(
                [
                    (
                        task,
                        category
                    )
                ]
                * count
            )

    expected_total = config[
        'benchmark'
    ]['total_pairs']

    if len(slots) != expected_total:
        raise ValueError(
            "Task/category slot count does not match total_pairs."
        )

    return slots



def generate_benchmark_plan(
    config: dict[str, Any],
) -> list[BenchmarkPlanRecord]:
    """Final benchmark generation planını oluşturur."""

    validate_benchmark_design(
        config
    )

    benchmark_config = config[
        "benchmark"
    ]

    source_language = benchmark_config[
        "languages"
    ]["source"]

    target_language = benchmark_config[
        "languages"
    ]["target"]

    total_pairs = benchmark_config[
        "total_pairs"
    ]

    seed = benchmark_config[
        "seed"
    ]

    answerability = config[
        "answerability"
    ]

    difficulty_pool = build_difficulty_pool(
        total_pairs,
        config["difficulty"],
    )

    task_category_slots = (
        build_task_category_slots(
            config
        )
    )

    random_generator = random.Random(
        seed
    )

    random_generator.shuffle(
        difficulty_pool
    )

    plan: list[
        BenchmarkPlanRecord
    ] = []

    task_indices: dict[
        str,
        int
    ] = {}

    for (
        task,
        category,
    ), difficulty in zip(
        task_category_slots,
        difficulty_pool,
        strict=True,
    ):
        task_indices[task] = (
            task_indices.get(
                task,
                0,
            )
            + 1
        )

        pair_id = (
            f"{task}_"
            f"{task_indices[task]:04d}"
        )

        plan.append(
            BenchmarkPlanRecord(
                pair_id=pair_id,
                task=task,
                category=category,
                difficulty=difficulty,
                source_language=source_language,
                target_language=target_language,
                is_answerable=answerability[
                    task
                ],
            )
        )

    if len(plan) != total_pairs:
        raise ValueError(
            "Generated benchmark plan size does not match total_pairs."
        )

    return plan


def save_benchmark_plan(
    records: list[BenchmarkPlanRecord],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    overwrite: bool = False,
) -> None:
    """Generation planını JSONL artifact olarak kaydeder."""

    if (
        output_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Output file already exists: {output_path}"
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
            json.dump(
                record.to_dict(),
                file,
                ensure_ascii=False,
            )

            file.write("\n")


def summarize_benchmark_plan(
    records: list[BenchmarkPlanRecord],
) -> dict[str, Any]:
    """Generation planının task/category/difficulty dağılımını özetler."""

    summary: dict[str, Any] = {
        "total_pairs": len(records),
        "by_task": {},
        "by_category": {},
        "by_difficulty": {},
        "answerable": 0,
        "unanswerable": 0,
    }

    for record in records:
        summary["by_task"][
            record.task
        ] = (
            summary["by_task"].get(
                record.task,
                0,
            )
            + 1
        )

        category_key = (
            f"{record.task}/"
            f"{record.category}"
        )

        summary["by_category"][
            category_key
        ] = (
            summary[
                "by_category"
            ].get(
                category_key,
                0,
            )
            + 1
        )

        summary["by_difficulty"][
            record.difficulty
        ] = (
            summary[
                "by_difficulty"
            ].get(
                record.difficulty,
                0,
            )
            + 1
        )

        if record.is_answerable:
            summary["answerable"] += 1
        else:
            summary["unanswerable"] += 1

    return summary


def main() -> None:
    """Final benchmark generation planını üretir ve kaydeder."""

    config = load_benchmark_design(
        DEFAULT_CONFIG_PATH
    )

    plan = generate_benchmark_plan(
        config
    )

    save_benchmark_plan(
        plan,
        DEFAULT_OUTPUT_PATH,
        overwrite=True,
    )

    summary = summarize_benchmark_plan(
        plan
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
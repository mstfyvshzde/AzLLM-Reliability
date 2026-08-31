"""Candidate benchmark dosyalarını topluca doğrular.

Bu modül data/candidates altındaki JSONL candidate dosyalarını yükler ve:

- kayıtların parse edilebilir olmasını
- item_id uniqueness
- pair_id + language uniqueness
- pair task consistency
- EN-AZ pair completeness
- task-specific metadata kurallarını
- category / difficulty uygunluğunu
- is_answerable constraint'lerini

tek bir validation akışı içinde kontrol eder.

Amaç review veya promotion aşamasından önce candidate benchmark verisinin
yapısal olarak temiz ve task specification'larıyla uyumlu olduğunu doğrulamaktır.
"""


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.data.build_benchmark import load_records
from src.data.pairing import (
    validate_complete_pairs,
    validate_pair_task_consistency,
)
from src.data.validate_benchmark import (
    validate_unique_item_ids,
    validate_unique_pair_languages,
)
from src.data.validate_task_metadata import (
    load_task_specifications,
    validate_task_metadata,
)


DEFAULT_CANDIDATES_DIR = Path(
    "data/candidates"
)

DEFAULT_TASK_SPECIFICATIONS_DIR = Path(
    "configs/tasks"
)

DEFAULT_REQUIRED_LANGUAGES = {
    "en",
    "az"
}



def discover_candidate_files(
    candidates_dir: Path
) -> list[Path]:
    """Candidate klasöründeki JSONL dosyalarını bulur, doğrular ve alfabetik sırayla döndürür.

    Kontroller:
    - candidates_dir mevcut mu?
    - verilen path gerçekten klasör mü?
    - klasör içinde JSONL dosyası var mı?

    Örnek klasör:
    data/candidates/
        reasoning.jsonl
        factual_qa.jsonl

    Fonksiyonun döndürdüğü sonuç:
    [
        Path("data/candidates/factual_qa.jsonl"),
        Path("data/candidates/reasoning.jsonl")
    ]

    Yani candidate benchmark dosyalarını toplar ve alfabetik sıraya koyar.

    Klasör yoksa FileNotFoundError,
    path klasör değilse veya JSONL dosyası bulunamazsa ValueError oluşturur.
    """

    if not candidates_dir.exists():
        raise FileNotFoundError(
            f"Candidates directory not found: {candidates_dir}"
        )

    if not candidates_dir.is_dir():
        raise ValueError(
            f"Candidates path is not a directory: {candidates_dir}"
        )

    candidate_files = sorted(
        path 
        for path in candidates_dir.glob(
            '*jsonl'
        )
        if path.is_file()
    )

    if not candidate_files:
        raise ValueError(
            f"No candidate JSONL files found in: {candidates_dir}"
        )

    return candidate_files




def get_task_names(
    records: list[Any]
) -> set[str]:
    """Candidate kayıtlarında bulunan benzersiz task adlarını toplar ve döndürür.

    Her record içindeki `task` alanını alır ve tekrar eden task isimlerini
    otomatik olarak kaldırmak için set içinde toplar.

    Örnek:
    records:
        task="reasoning"
        task="reasoning"
        task="factual_qa"

    Sonuç:
    {
        "reasoning",
        "factual_qa"
    }

    Hiç task bulunamazsa ValueError oluşturur.
    """

    task_names = {
        record.task
        for record in records
    }

    if not task_names:
        raise ValueError(
            "Candidate records contain no tasks."
        )

    return task_names



def load_task_specification_for_task(
    task: str,
    specifications_dir: Path
) -> dict[str, Any]:
    """Belirtilen task için ilgili YAML specification dosyasını yükler.

    Task adına göre specification dosya yolunu oluşturur ve
    `load_task_specifications()` ile config içeriğini yükler.

    Örnek:
    task = "reasoning"
    specifications_dir = Path("configs/tasks")

    Oluşan path:
    configs/tasks/reasoning.yaml

    Sonuç:
    reasoning task'ına ait specification dictionary'si döndürülür.
    """

    specification_path = (
        specifications_dir 
        / f'{task}.yaml'
    )

    return load_task_specifications(specification_path)



def validate_task_records(
    records: list[Any],
    task_specifications: dict[
        str,
        dict[str, Any]
    ]
) -> None:
    """Her candidate record'u kendi task specification'ına göre doğrular."""


    for record in records:
        if record.task not in task_specifications:
            raise ValueError(
                f"No task specification loaded for task '{record.task}'."
            )

        validate_task_metadata(
            record,
            task_specifications[
                record.task
            ]
        )


def validate_candidate_records(
    records: list[Any],
    specifications_dir: Path,
    required_languages: set[str]
) -> None:
    """Candidate record listesinin tüm validation kurallarını çalıştırır."""

    if not records:
        raise ValueError(
            "Candidate records cannot be empty."
        )

    validate_unique_item_ids(
        records
    )

    validate_unique_pair_languages(
        records
    )

    validate_pair_task_consistency(
        records
    )

    validate_complete_pairs(
        records,
        required_languages
    )

    task_names = get_task_names(
        records
    )

    task_specifications = {
        task: load_task_specification_for_task(
            task=task,
            specifications_dir=specifications_dir
        )
        for task in task_names
    }

    validate_task_records(
        records=records,
        task_specifications=task_specifications
    )


def validate_candidate_file(
    candidate_path: Path,
    specifications_dir: Path,
    required_languages: set[str]
) -> int:
    """Tek bir candidate JSONL dosyasını doğrular ve record sayısını döndürür."""

    records = load_records(
        candidate_path
    )

    validate_candidate_records(
        records=records,
        specifications_dir=specifications_dir,
        required_languages=required_languages,
    )

    return len(
        records
    )


def validate_all_candidates(
    candidates_dir: Path = DEFAULT_CANDIDATES_DIR,
    specifications_dir: Path = DEFAULT_TASK_SPECIFICATIONS_DIR,
    required_languages: set[str] | None = None
) -> dict[str, int]:
    """Candidate klasöründeki tüm JSONL dosyalarını doğrular.

    Dönen dictionary her dosya için doğrulanan record sayısını içerir.

    Örnek:
        {
            "reasoning.jsonl": 10,
            "unanswerable.jsonl": 10,
        }
    """

    if required_languages is None:
        required_languages = set(
            DEFAULT_REQUIRED_LANGUAGES
        )

    candidate_files = discover_candidate_files(
        candidates_dir
    )

    results: dict[str, int] = {}

    for candidate_path in candidate_files:
        record_count = validate_candidate_file(
            candidate_path=candidate_path,
            specifications_dir=specifications_dir,
            required_languages=required_languages
        )

        results[
            candidate_path.name
        ] = record_count

    return results



def parse_arguments() -> argparse.Namespace:
    """Command-line argumentlerini parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate all candidate benchmark JSONL files."
        )
    )

    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=DEFAULT_CANDIDATES_DIR,
        help="Candidate JSONL directory."
    )

    parser.add_argument(
        "--specifications-dir",
        type=Path,
        default=DEFAULT_TASK_SPECIFICATIONS_DIR,
        help="Task specification YAML directory."
    )

    return parser.parse_args()



def main() -> None:
    """Candidate validation CLI entry point."""

    args = parse_arguments()

    results = validate_all_candidates(
        candidates_dir=args.candidates_dir,
        specifications_dir=args.specifications_dir,
    )

    total_records = sum(
        results.values()
    )

    for filename, record_count in results.items():
        print(
            f"PASS {filename}: {record_count} records"
        )

    print(
        f"Validated {len(results)} candidate files "
        f"with {total_records} total records."
    )


if __name__ == "__main__":
    main()
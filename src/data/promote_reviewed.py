"""Review edilmiş benchmark kayıtlarını final raw benchmark'a aktarır.

Bu modül reviewed JSONL dosyalarını yükler, yalnızca tamamen onaylanmış
pair'leri seçer ve benchmark builder tarafından kullanılacak final
raw JSONL dosyasını oluşturur.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.benchmark_record import BenchmarkRecord
from src.data.build_benchmark import load_records
from src.data.pairing import (
    validate_complete_pairs,
    validate_pair_task_consistency
)
from src.data.review_candidates import get_approved_records
from src.data.validate_benchmark import (
    validate_unique_item_ids,
    validate_unique_pair_languages
)
from src.data.validate_task_metadata import (
    load_task_specifications,
    validate_task_metadata,
)

def collect_reviewed_records(
    input_dir: Path
) -> list[BenchmarkRecord]:
    """Reviewed klasöründeki tüm JSONL benchmark kayıtlarını toplar.

    Verilen klasördeki JSONL dosyalarını bulur, her dosyadaki BenchmarkRecord
    kayıtlarını yükler ve hepsini tek bir liste içinde birleştirir.

    Örnek:
        data/reviewed/
            reasoning.jsonl  → 10 kayıt
            factual_qa.jsonl → 8 kayıt
        
        Sonuç:
            records → toplam 18 BenchmarkRecord

    Klasör bulunamazsa FileNotFoundError, klasörde JSONL dosyası yoksa
    ValueError oluşturur.
    """

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Reviewed directory not found: {input_dir}"
        )

    input_paths = sorted(input_dir.glob('*.jsonl'))

    if not input_paths:
        raise ValueError(
            f"No reviewed JSONL files found in: {input_dir}"
        )

    records: list[BenchmarkRecord] = []

    for input_path in input_paths:
        records.extend(load_records(input_path))

    return records


def validate_promoted_records(
    records: list[BenchmarkRecord],
    task_specifications: dict[str, dict]
) -> None:
    """Final benchmark'a aktarılacak promoted kayıtların bütünlüğünü doğrular.

    Promoted records, manuel review'dan geçip onaylanan ve final benchmark'a
    aktarılmaya hazır EN-AZ kayıtlarıdır.

    Örnek:
        reasoning_001_en → approved
        reasoning_001_az → approved
        → promoted records → final benchmark

    Kontroller:
        - Kayıt listesi boş mu?
        - Aynı item_id tekrar ediyor mu?
        - Aynı pair_id-language kombinasyonu tekrar ediyor mu?
        - Her pair hem "en" hem "az" kaydına sahip mi?
        - Aynı pair içindeki kayıtlar aynı task'a mı ait?
        - Her kayıt kendi task-specific metadata kurallarına uyuyor mu?

    Kurallardan biri ihlal edilirse ValueError oluşturur.
    """

    if not records:
        raise ValueError("No approved records available for promotion.")

    validate_unique_item_ids(records)
    validate_unique_pair_languages(records)
    validate_complete_pairs(records, {'en', 'az'})
    validate_pair_task_consistency(records)

    for record in records:
        task_specification = task_specifications.get(record.task)

        if task_specification is None:
            raise ValueError(
                f"Task specification not loaded for '{record.task}'."
            )

        validate_task_metadata(
            record,
            task_specification,
        )


def save_final_records(
    records: list[BenchmarkRecord],
    output_path: Path
) -> None:
    """Final benchmark kayıtlarını tek JSONL dosyasına yazar."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as file:
        for record in records:
            file.write(
                json.dumps(
                    record.to_dict(),
                    ensure_ascii=False
                )
                + '\n'
            )


def parse_arguments() -> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştırır."""

    parser = argparse.ArgumentParser(
        description="Approved benchmark pair'lerini final raw veriye aktarır."
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path("data/reviewed"),
        help="Reviewed JSONL dosyalarının bulunduğu klasör.",
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/raw/benchmark.jsonl"),
        help="Final raw benchmark JSONL dosyasının yolu.",
    )

    return parser.parse_args()


def main() -> None:
    """Reviewed-to-raw promotion sürecini çalıştırır."""
    args = parse_arguments()

    records = collect_reviewed_records(args.input_dir)
    approved_records = get_approved_records(records)

    task_names = {
        record.task
        for record in approved_records
    }

    task_specifications = {
        task_name: load_task_specifications(
            Path("configs/tasks") / f"{task_name}.yaml"
        )
        for task_name in task_names
    }

    validate_promoted_records(
        approved_records,
        task_specifications,
    )

    save_final_records(approved_records, args.output)

    print(
        f"Promoted {len(approved_records)} approved records "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
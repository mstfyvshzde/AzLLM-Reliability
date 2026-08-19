"""Eşleştirilmiş İngilizce-Azerbaycanca benchmark veri kümesini oluşturur.

Bu modül benchmark oluşturma ayarlarını yükler, ham benchmark kayıtlarını
doğrular, iki dil arasındaki eşleştirme yapısını korur ve sonraki değerlendirme
aşamalarında kullanılacak standartlaştırılmış benchmark çıktıları üretir.
"""


# type hint'lerin daha esnek çalışmasını sağlar.
from __future__ import annotations

import json
import argparse 
from pathlib import Path
from typing import Any

import yaml


from src.data.benchmark_record import BenchmarkRecord
from src.data.pairing import (
    validate_complete_pairs,
    validate_pair_task_consistency
)

from src.data.validate_benchmark import (
    validate_record,
    validate_unique_item_ids,
    validate_unique_pair_languages
)

from src.data.task_registry import (
    get_enabled_tasks,
    load_task_config,
    validate_task
)


def load_config(config_path: Path) -> dict[str, Any]:
    """Benchmark ayarlarını içeren YAML dosyasını yükler.

    Args:
        config_path: Örneğin `configs/benchmark.yaml` dosyasının yolu.

    Returns:
        YAML içeriğini Python sözlüğü olarak döndürür.

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
        ValueError: Dosya geçerli bir YAML mapping değilse.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Benchmark config must contain a YAML mapping.")

    return config



def load_records(input_path: Path) -> list[BenchmarkRecord]:
    """JSONL dosyasındaki ham benchmark kayıtlarını yükler.

    Dosyayı satır satır okur, her JSON satırını bir BenchmarkRecord nesnesine
    dönüştürür ve oluşturulan tüm kayıtları `records` listesinde toplar.

    Örnek:
        JSONL satırı:
        {"item_id": "reasoning_001_en", "pair_id": "reasoning_001",
        "language": "en", "question": "What is 2 + 2?",
        "reference_answer": "4"}

        records:
        [
            BenchmarkRecord(
                item_id="reasoning_001_en",
                pair_id="reasoning_001",
                language="en",
                question="What is 2 + 2?",
                reference_answer="4"
            )
        ]

    Dosya bulunamazsa FileNotFoundError, bozuk veya eksik bir kayıt varsa
    ValueError oluşturur.
    """

    if not input_path.exists():
        raise FileNotFoundError (f"Input file not found: {input_path}")

    records: list[BenchmarkRecord] = []

    with input_path.open('r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                
                record = BenchmarkRecord(
                    item_id=data["item_id"],
                    pair_id=data["pair_id"],
                    language=data["language"],
                    task=data["task"],
                    question=data["question"],
                    reference_answer=data["reference_answer"],
                    metadata=data.get("metadata", {}),
                )

            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid benchmark record at line {line_number}: {error}"
                ) from error

            records.append(record)

    return records



def validate_records(
    records: list[BenchmarkRecord],
    config: dict[str, Any],
    enabled_tasks: set[str]
) -> None:
    """Tüm benchmark kayıtlarını YAML config kurallarına göre doğrular.

    Önce source ve target dillerini config dosyasından alır. Ardından her kaydın
    temel alanlarını ve dilini kontrol eder.

    Config içinde etkinleştirilmişse ayrıca:
    - tekrar eden item_id değerlerini,
    - tekrar eden pair_id-language kombinasyonlarını,
    - her pair için hem İngilizce hem Azerbaycanca kaydın bulunmasını kontrol eder.

    Kurallardan biri ihlal edilirse ValueError oluşturur.
    """

    required_languages = {
        config["languages"]["source"],
        config["languages"]["target"],
    }

    for record in records:
        validate_record(record, required_languages)
        # enabled_tasks → aktif olan task'ların listesi/set'i.
        validate_task(record.task, enabled_tasks)

    if config["validation"]["reject_duplicate_item_ids"]:
        validate_unique_item_ids(records)

    if config["validation"]["reject_duplicate_pair_language"]:
        validate_unique_pair_languages(records)

    if config["pairing"]["require_both_languages"]:
        validate_complete_pairs(records, required_languages)
        validate_pair_task_consistency(records)


def save_records(
    records: list[BenchmarkRecord],
    output_dir: Path
) -> None:
    """Doğrulanmış kayıtları dillere göre ayrı JSONL dosyalarına kaydeder."""

    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_language: dict[str, list[BenchmarkRecord]] = {}

    for record in records:
        records_by_language.setdefault(record.language, []).append(record)

    for language, language_records in records_by_language.items():
        language_dir = output_dir / language
        language_dir.mkdir(parents=True, exist_ok=True)

        output_path = language_dir / 'benchmark.jsonl'

        with output_path.open('w', encoding='utf-8') as file:
            for record in language_records:
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
        description="Eşleştirilmiş İngilizce-Azerbaycanca benchmark oluşturur."
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path("configs/benchmark.yaml"),
        help="Benchmark yapılandırma dosyasının yolu.",
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Ham benchmark JSONL dosyasının yolu.'
    )

    return parser.parse_args()



def main() -> None:
    """Benchmark oluşturma sürecinin ana giriş noktasını çalıştırır."""

    args = parse_arguments()

    config = load_config(args.config)
    task_config = load_task_config(Path("configs/tasks.yaml"))
    enabled_tasks = get_enabled_tasks(task_config)
    records = load_records(args.input)

    validate_records(records, config, enabled_tasks)

    output_dir = Path(config["paths"]["benchmark_dir"])

    english_output = output_dir / "en" / "benchmark.jsonl"
    azerbaijani_output = output_dir / "az" / "benchmark.jsonl"

    if (
        not config["experiment"]["overwrite_outputs"]
        and (english_output.exists() or azerbaijani_output.exists())
    ):
        raise FileExistsError(
            "Benchmark output files already exist. "
            "Enable overwrite_outputs to replace them."
        )

    save_records(records, output_dir)

    print(
        f"Built benchmark with {len(records)} records "
        f"at {output_dir}"
    )


if __name__ == "__main__":
    main()



"""Paired EN-AZ benchmark candidate kayıtlarını oluşturur.

Bu modül generation config içindeki task ve identifier kurallarını kullanarak
manuel inceleme öncesi benchmark candidate kayıtlarının temel yapısını üretir.
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.data.benchmark_record import BenchmarkRecord
from src.data.task_registry import(
    get_enabled_tasks,
    load_task_config
)


def load_generation_config(config_path: Path) -> dict[str, Any]:
    """Candidate generation ayarlarını içeren YAML dosyasını yükler ve doğrular.

    Örnek generation config:

    generation:
        languages: ["en", "az"]

    identifiers:
        pair_id_format: "{task}_{index:03d}"
        item_id_format: "{pair_id}_{language}"

    Fonksiyon bu YAML içeriğini Python sözlüğüne dönüştürür ve özellikle
    `generation` bölümünün mevcut olup olmadığını kontrol eder.

    Dosya bulunamazsa FileNotFoundError, içerik geçerli bir mapping değilse
    veya `generation` bölümü yoksa ValueError oluşturur.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Generation config not found: {config_path}"
        )

    with config_path.open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Generation config must contain a YAML mapping."
        )

    if 'generation' not in config:
        raise ValueError(
            "Generation config must contain a 'generation' section."
        )

    return config


def create_candidate_pair(
    task: str,
    index: int,
    question_en: str,
    question_az: str,
    reference_answer_en: str,
    reference_answer_az: str,
    config: dict[str, Any]
) -> list[BenchmarkRecord]:
    """Tek bir semantic İngilizce-Azerbaycanca candidate pair oluşturur.

    Aynı görevin EN ve AZ sürümleri için ortak bir pair_id üretir, her dil için
    ayrı item_id oluşturur ve iki BenchmarkRecord kaydını bir liste içinde döndürür.

    Örnek:
        task = "reasoning"
        index = 1

        pair_id:
        "reasoning_001"

        Oluşan kayıtlar:
        reasoning_001_en → language="en"
        reasoning_001_az → language="az"

    Her iki kayıt aynı pair_id değerini paylaşır, ancak question ve
    reference_answer alanları kendi dillerine aittir. Yeni kayıtların
    review_status değeri başlangıçta "pending" olarak ayarlanır.
    """

    # Config'teki pair_id formatını task ve index ile doldurur.
    # Örnek: "{task}_{index:03d}" → "reasoning_001"
    pair_id = config['identifiers']['pair_id_format'].format(
        task=task,
        index=index
    )

    records: list[BenchmarkRecord] = []

    values = {
        'en': (question_en, reference_answer_en),
        'az': (question_az, reference_answer_az)
    }

    for language in config['generation']['languages']:
        question, reference_answer = values[language]

        item_id = config['identifiers']['item_id_format'].format(
            pair_id=pair_id,
            language=language
        )

        records.append(
            BenchmarkRecord(
                item_id=item_id,
                pair_id=pair_id,
                language=language,
                task=task,
                question=question,
                reference_answer=reference_answer,
                metadata={
                    "review_status": "pending",
                },
            )
        )

    return records



def save_candidates(
    records: list[BenchmarkRecord],
    output_path: Path
) -> None:
    """Candidate kayıtlarını JSONL dosyasına kaydeder."""

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
        description="Paired EN-AZ benchmark candidate oluşturur."
    )

    parser.add_argument(
        '--generation-config',
        type=Path,
        default=Path("configs/generation.yaml"),
        help="Generation config dosyasının yolu."
    )

    parser.add_argument(
        '--task-config',
        type=Path,
        default=Path("configs/tasks.yaml"),
        help="Task config dosyasının yolu."
    )

    return parser.parse_args()



def main() -> None:
    """Candidate generation giriş noktasını çalıştırır."""
    args = parse_arguments()

    generation_config = load_generation_config(
        args.generation_config
    )

    task_config = load_task_config(args.task_config)
    enabled_tasks = get_enabled_tasks(task_config)

    print(
        "Candidate generator ready for tasks: "
        f"{sorted(enabled_tasks)}"
    )


if __name__ == "__main__":
    main()
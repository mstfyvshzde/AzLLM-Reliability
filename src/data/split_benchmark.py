"""Final benchmark kayıtlarını reproducible train-dev-test split'lerine ayırır.

Bu modül benchmark kayıtlarını pair_id seviyesinde gruplar ve aynı EN-AZ
semantic pair içindeki kayıtların farklı split'lere düşmesini engeller.

Split işlemi deterministic bir random seed kullanır. Böylece aynı benchmark,
aynı split oranları ve aynı seed ile tekrar çalıştırıldığında aynı train,
development ve test bölümleri üretilir.
"""


from __future__ import annotations

import argparse
import random 
from pathlib import Path
from typing import Any

from src.data.benchmark_record import BenchmarkRecord
from src.data.build_benchmark import load_records
from src.data.pairing import (
    group_by_pair,
    validate_complete_pairs,
    validate_pair_task_consistency
)
from src.data.promote_reviewed import save_final_records


def validate_split_ratios(
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float
) -> None:
    """Train, dev ve test split oranlarının geçerli olduğunu doğrular.

    Her oran 0 ile 1 arasında olmalıdır ve üç oranın toplamı 1.0 olmalıdır.

    Örnek:
        train_ratio = 0.70
        dev_ratio = 0.15
        test_ratio = 0.15

    Geçersiz oranlar için ValueError oluşturur.
    """

    ratios = {
        "train": train_ratio,
        "dev": dev_ratio,
        "test": test_ratio
    }

    for name, ratio in ratios.items():
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                f"{name} ratio must be between 0 and 1, got {ratio}."
            )

    total = train_ratio + dev_ratio + test_ratio

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Train, dev and test ratios must sum to 1.0. "
            f"Got {total:.6f}."
        )


def calculate_split_sizes(
    pair_count: int,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float
) -> tuple[int, int, int]:
    """Toplam pair sayısından train, dev ve test pair sayılarını hesaplar.

    Önce train ve dev boyutları hesaplanır. Kalan pair'ler test split'ine
    atanarak toplam pair sayısının kesin olarak korunması sağlanır.

    Küçük benchmark'larda rounding nedeniyle bazı split'ler boş kalabilir.
    """

    validate_split_ratios(
        train_ratio,
        dev_ratio,
        test_ratio
    )

    if pair_count < 0:
        raise ValueError(
            f"Pair count cannot be negative: {pair_count}"
        )

    train_size = int(pair_count * train_ratio)
    dev_size = int(pair_count * dev_ratio)
    test_size = pair_count - (train_size + dev_size)

    return train_size, dev_size, test_size


def split_records_by_pair(
    records: list[BenchmarkRecord],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int
) -> dict[str, list[BenchmarkRecord]]:
    """Benchmark kayıtlarını pair-aware train-dev-test split'lerine ayırır.

    Split işlemi individual record seviyesinde değil pair_id seviyesinde yapılır.
    Böylece aynı semantic görevin İngilizce ve Azerbaycanca sürümleri her zaman
    aynı split içinde kalır.

    Örnek:
        reasoning_0001_en
        reasoning_0001_az

    Bu iki kayıt birlikte train, dev veya test split'lerinden yalnızca birine
    atanabilir.

    Pair sırası önce deterministic olarak sıralanır, ardından verilen seed ile
    shuffle edilir. Bu yaklaşım input dosyasındaki record sırasından bağımsız
    reproducible split üretir.
    """

    if not records:
        raise ValueError(
            "Cannot split an empty benchmark."
        )

    validate_split_ratios(
        train_ratio,
        dev_ratio,
        test_ratio
    )

    validate_complete_pairs(
        records,
        {'en', 'az'}
    )

    validate_pair_task_consistency(records)

    grouped_records = group_by_pair(records)

    pair_ids = sorted(grouped_records)

    random_generator = random.Random(seed)
    random_generator.shuffle(pair_ids)

    train_size, dev_size, _ = calculate_split_sizes(
        pair_count=len(pair_ids),
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio
    )

    train_end = train_size
    dev_end = train_size + dev_size

    train_pair_ids = pair_ids[:train_end]
    dev_paiir_ids = pair_ids[train_end:dev_end]
    test_pair_ids = pair_ids[dev_end:]

    splits: dict[str, list[BenchmarkRecord]] = {
        'train': [],
        'dev': [],
        'test': []
    }

    for pair_id in train_pair_ids:
        splits['train'].extend(
            grouped_records[pair_id]
        )

    for pair_id in dev_paiir_ids:
        splits['dev'].extend(
            grouped_records[pair_id]
        )

    for pair_id in test_pair_ids:
        splits['test'].extend(
            grouped_records[pair_id]
        )

    return splits


def validate_split_integrity(
    original_records: list[BenchmarkRecord],
    splits: dict[str, list[BenchmarkRecord]]
) -> None:
    """Oluşturulan split'lerin veri bütünlüğünü koruduğunu doğrular.

    `splits`, benchmark kayıtlarının train, dev ve test olarak ayrılmış halidir.

    Örnek splits:
        {
            "train": [record_001_en, record_001_az],
            "dev":   [record_002_en, record_002_az],
            "test":  [record_003_en, record_003_az],
        }

    Burada aynı pair'in EN ve AZ kayıtları aynı split içinde kalmalıdır.

    Kontroller:
        - Train, dev ve test split'leri mevcut mu?
        - Her original record tam olarak bir split içinde mi?
        - Aynı pair_id farklı split'lere dağılmış mı?
        - Kayıt kaybı veya duplicate kayıt oluşmuş mu?

    Kurallardan biri ihlal edilirse ValueError oluşturur.
    """

    required_split_names = {
        'train',
        'dev',
        'test'
    }

    if set(splits) != required_split_names:
        raise ValueError(
            "Splits must contain exactly: "
            f"{sorted(required_split_names)}"
        )

    original_item_ids = {
        record.item_id
        for record in original_records
    }

    split_item_ids: list[str] = []
    pair_locations: dict[str, str] = {}

    for split_name, split_records in splits.items():
        for record in split_records:
            split_item_ids.append(
                record.item_id
            )

            existing_split = pair_locations.get(
                record.pair_id
            )

            if (
                existing_split is not None
                and existing_split != split_name
            ): 
                raise ValueError(
                    f"Pair '{record.pair_id}' appears in both "
                    f"'{existing_split}' and '{split_name}' splits."
                )

            pair_locations[record.pair_id] = split_name

    if len(split_item_ids) != len(set(split_item_ids)):
        raise ValueError(
            "Duplicate benchmark records detected across splits."
        )

    if set(split_item_ids) != original_item_ids:
        missing = original_item_ids - set(split_item_ids)
        unexpected = set(split_item_ids) - original_item_ids

        raise ValueError(
            "Split records do not match original benchmark records. "
            f"Missing: {sorted(missing)}. "
            f"Unexpected: {sorted(unexpected)}."
        )



def save_splits(
    splits: dict[str, list[BenchmarkRecord]],
    output_dir: Path
) -> None:
    """Train, dev ve test split'lerini ayrı JSONL dosyalarına kaydeder.

    `splits` içindeki her split için ayrı bir dosya oluşturur ve kayıtları
    ilgili JSONL dosyasına yazar.

    Örnek splits:
        {
            "train": [record_001_en, record_001_az],
            "dev":   [record_002_en, record_002_az],
            "test":  [record_003_en, record_003_az],
        }

    Örnek çıktı:
        data/splits/
            train.jsonl → record_001_en, record_001_az
            dev.jsonl   → record_002_en, record_002_az
            test.jsonl  → record_003_en, record_003_az
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for split_name, records in splits.items():
        output_path = output_dir / f'{split_name}.jsonl'

        save_final_records(
            records,
            output_path
        )



def parse_arguments()-> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştırır."""

    parser = argparse.ArgumentParser(
        description=(
            "Final benchmark kayıtlarını pair-aware "
            "train-dev-test split'lerine ayırır."
        )
    )

    parser.add_argument(
        '--input',
        type=Path,
        default=Path("data/raw/benchmark.jsonl"),
        help="Split edilecek final raw benchmark JSONL dosyası."
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("data/processed/splits"),
        help="Train, dev ve test split dosyalarının yazılacağı klasör."
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help="Train split oranı."
    )

    parser.add_argument(
        '--dev-ratio',
        type=float,
        default=0.15,
        help="Development split oranı.",
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help="Test split oranı."
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=17,
        help="Reproducible split için random seed.",
    )

    return parser.parse_args()



def main() -> None:
    """Benchmark splitting sürecinin ana giriş noktasını çalıştırır."""
    args = parse_arguments()

    records = load_records(
        args.input
    )

    splits = split_records_by_pair(
        records=records,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    validate_split_integrity(
        records,
        splits,
    )

    save_splits(
        splits,
        args.output_dir,
    )

    print(
        "Benchmark split completed. "
        f"Train records: {len(splits['train'])}, "
        f"Dev records: {len(splits['dev'])}, "
        f"Test records: {len(splits['test'])}."
    )


if __name__ == "__main__":
    main()
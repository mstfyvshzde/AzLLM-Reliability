"""Benchmark candidate kayıtlarının manuel review sürecini yönetir.

Bu modül candidate JSONL kayıtlarını yükler, aynı pair içindeki kayıtların
review durumunu günceller ve review sonucunu reviewed veri klasörüne yazar.
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.benchmark_record import BenchmarkRecord
from src.data.build_benchmark import load_records
from src.data.pairing import (
    group_by_pair,
    validate_complete_pairs,
    validate_pair_task_consistency,
)


# Bu dosyada benchmark → review'dan geçen EN-AZ soru çiftleri yönetilir.
# Örnek:
# reasoning_001_en → "What is 2 + 3?" → approved
# reasoning_001_az → "2 + 3 neçə edir?" → approved
# pending  -> Henüz manuel olarak incelenmedi.
# approved -> İncelendi ve benchmark için kabul edildi.
# rejected -> İncelendi ve benchmark için reddedildi.
VALID_REVIEW_STATUSES = {
    "pending",
    "approved",
    "rejected",
}


def validate_review_statuses(
    records: list[BenchmarkRecord],
) -> None:
    """Tüm kayıtların geçerli bir review_status değerine sahip olduğunu doğrular.

    Her BenchmarkRecord metadata alanında bir review_status değeri taşımalıdır.
    Bu değer yalnızca VALID_REVIEW_STATUSES içinde tanımlanan değerlerden biri
    olabilir.

    Örnek geçerli değerler:
        pending
        approved
        rejected

    review_status eksik veya geçersizse ValueError oluşturur.
    """

    for record in records:
        status = record.metadata.get("review_status")

        if status not in VALID_REVIEW_STATUSES:
            raise ValueError(
                f"Invalid review status '{status}' "
                f"for item '{record.item_id}'. "
                f"Expected one of: {sorted(VALID_REVIEW_STATUSES)}"
            )


def set_pair_review_status(
    records: list[BenchmarkRecord],
    pair_id: str,
    status: str,
) -> list[BenchmarkRecord]:
    """Belirtilen pair içindeki tüm kayıtların review durumunu günceller.

    Verilen pair_id ile eşleşen EN ve AZ kayıtlarını bulur ve metadata içindeki
    review_status değerini yeni status ile değiştirir.

    Örnek:
        pair_id = "reasoning_001"
        status = "approved"

        Önce:
        reasoning_001_en → review_status="pending"
        reasoning_001_az → review_status="pending"

        Sonra:
        reasoning_001_en → review_status="approved"
        reasoning_001_az → review_status="approved"

    Geçersiz bir status verilirse veya pair_id bulunamazsa ValueError oluşturur.
    """

    if status not in VALID_REVIEW_STATUSES:
        raise ValueError(
            f"Invalid review status '{status}'. "
            f"Expected one of: {sorted(VALID_REVIEW_STATUSES)}"
        )

    updated_records: list[BenchmarkRecord] = []
    pair_found = False

    for record in records:
        if record.pair_id != pair_id:
            updated_records.append(record)
            continue

        pair_found = True

        metadata = dict(record.metadata)
        metadata["review_status"] = status

        updated_records.append(
            BenchmarkRecord(
                item_id=record.item_id,
                pair_id=record.pair_id,
                language=record.language,
                task=record.task,
                question=record.question,
                reference_answer=record.reference_answer,
                metadata=metadata,
            )
        )

    if not pair_found:
        raise ValueError(
            f"Pair not found: {pair_id}"
        )

    return updated_records


def get_approved_records(
    records: list[BenchmarkRecord],
) -> list[BenchmarkRecord]:
    """Yalnızca tamamen onaylanmış pair'leri döndürür.

    Bir pair yalnızca pair içindeki bütün kayıtların review_status değeri
    "approved" olduğunda final approved kayıtları arasına alınır.

    Böylece EN kaydı approved fakat AZ kaydı pending veya rejected olan
    kısmi pair'lerin benchmark'a ilerlemesi engellenir.
    """

    approved_records: list[BenchmarkRecord] = []

    for pair_records in group_by_pair(records).values():
        statuses = {
            record.metadata.get("review_status")
            for record in pair_records
        }

        if statuses == {"approved"}:
            approved_records.extend(pair_records)

    return approved_records


def save_records(
    records: list[BenchmarkRecord],
    output_path: Path,
) -> None:
    """Review edilmiş kayıtları JSONL dosyasına kaydeder."""

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
    """Komut satırı argümanlarını ayrıştırır."""

    parser = argparse.ArgumentParser(
        description="Benchmark candidate pair review işlemini yürütür."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Candidate JSONL dosyasının yolu.",
    )

    parser.add_argument(
        "--pair-id",
        required=True,
        help="Review edilecek pair kimliği.",
    )

    parser.add_argument(
        "--status",
        choices=sorted(VALID_REVIEW_STATUSES),
        required=True,
        help="Pair için atanacak review durumu.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Review sonucunun yazılacağı JSONL dosyası.",
    )

    return parser.parse_args()


def main() -> None:
    """Candidate review sürecinin ana giriş noktasını çalıştırır."""
    args = parse_arguments()

    records = load_records(args.input)

    validate_complete_pairs(
        records,
        {"en", "az"},
    )

    validate_pair_task_consistency(records)
    validate_review_statuses(records)

    reviewed_records = set_pair_review_status(
        records,
        args.pair_id,
        args.status,
    )

    validate_review_statuses(reviewed_records)

    save_records(
        reviewed_records,
        args.output,
    )

    approved_records = get_approved_records(
        reviewed_records
    )

    print(
        f"Reviewed pair '{args.pair_id}' as '{args.status}'. "
        f"Reviewed records written: {len(reviewed_records)}. "
        f"Approved records available: {len(approved_records)}."
    )


if __name__ == "__main__":
    main()
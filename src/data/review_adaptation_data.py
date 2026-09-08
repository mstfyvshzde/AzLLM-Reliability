"""Adaptation datası için human-review yardımcılarını sağlar.

Bu modül generated adaptation kayıtlarının review durumunu yönetir.

Geçerli review durumları:

    pending
    approved
    rejected

Review işlemi training datasını doğrudan değiştirmek yerine
metadata.review_status alanını günceller.

Yalnızca approved kayıtlar final adaptation dataset'e alınmalıdır.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALID_REVIEW_STATUSES = {
    "pending",
    "approved",
    "rejected",
}


def load_records(
    path: Path,
) -> list[dict[str, Any]]:
    """Adaptation JSONL kayıtlarını yükler."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}."
                ) from error

            if not isinstance(
                row,
                dict,
            ):
                raise ValueError(
                    f"Line {line_number} must be a JSON object."
                )

            rows.append(
                row
            )

    return rows


def validate_review_status(
    status: str,
) -> None:
    """Review status değerini doğrular."""

    if status not in VALID_REVIEW_STATUSES:
        raise ValueError(
            f"Invalid review status: {status}. "
            f"Expected one of "
            f"{sorted(VALID_REVIEW_STATUSES)}."
        )


def set_review_status(
    records: list[dict[str, Any]],
    item_id: str,
    status: str,
) -> list[dict[str, Any]]:
    """Tek bir adaptation kaydının review status'unu değiştirir."""

    validate_review_status(
        status
    )

    found = False

    updated_records: list[
        dict[str, Any]
    ] = []

    for record in records:
        updated = dict(
            record
        )

        if record.get(
            "item_id"
        ) == item_id:
            metadata = dict(
                record.get(
                    "metadata",
                    {},
                )
            )

            metadata[
                "review_status"
            ] = status

            updated[
                "metadata"
            ] = metadata

            found = True

        updated_records.append(
            updated
        )

    if not found:
        raise ValueError(
            f"Unknown item_id: {item_id}"
        )

    return updated_records


def set_all_review_status(
    records: list[dict[str, Any]],
    status: str,
) -> list[dict[str, Any]]:
    """Bütün kayıtların review status'unu değiştirir."""

    validate_review_status(
        status
    )

    updated_records: list[
        dict[str, Any]
    ] = []

    for record in records:
        updated = dict(
            record
        )

        metadata = dict(
            record.get(
                "metadata",
                {},
            )
        )

        metadata[
            "review_status"
        ] = status

        updated[
            "metadata"
        ] = metadata

        updated_records.append(
            updated
        )

    return updated_records


def get_records_by_status(
    records: list[dict[str, Any]],
    status: str,
) -> list[dict[str, Any]]:
    """Belirli review durumundaki kayıtları döndürür."""

    validate_review_status(
        status
    )

    return [
        record
        for record in records
        if record.get(
            "metadata",
            {},
        ).get(
            "review_status"
        )
        == status
    ]


def get_approved_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sadece approved adaptation kayıtlarını döndürür."""

    return get_records_by_status(
        records=records,
        status="approved",
    )


def build_review_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Review status dağılımını özetler."""

    counts = {
        status: 0
        for status in sorted(
            VALID_REVIEW_STATUSES
        )
    }

    for record in records:
        status = record.get(
            "metadata",
            {},
        ).get(
            "review_status"
        )

        if status not in VALID_REVIEW_STATUSES:
            raise ValueError(
                f"Invalid or missing review_status for "
                f"{record.get('item_id', '<unknown>')}: "
                f"{status}"
            )

        counts[
            status
        ] += 1

    return {
        "total": len(
            records
        ),
        "approved": counts[
            "approved"
        ],
        "rejected": counts[
            "rejected"
        ],
        "pending": counts[
            "pending"
        ],
    }


def save_records(
    records: list[dict[str, Any]],
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """Adaptation kayıtlarını JSONL olarak kaydeder."""

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
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_arguments() -> argparse.Namespace:
    """CLI argumentlerini parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Review Azerbaijani adaptation records."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--item-id",
        type=str,
    )

    parser.add_argument(
        "--status",
        choices=sorted(
            VALID_REVIEW_STATUSES
        ),
    )

    parser.add_argument(
        "--approve-all",
        action="store_true",
    )

    parser.add_argument(
        "--export-approved",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_arguments()

    records = load_records(
        args.input
    )

    if args.approve_all:
        records = set_all_review_status(
            records=records,
            status="approved",
        )

    elif (
        args.item_id is not None
        or args.status is not None
    ):
        if (
            args.item_id is None
            or args.status is None
        ):
            raise ValueError(
                "--item-id and --status "
                "must be provided together."
            )

        records = set_review_status(
            records=records,
            item_id=args.item_id,
            status=args.status,
        )

    if args.export_approved:
        records = get_approved_records(
            records
        )

    save_records(
        records=records,
        output_path=args.output,
        overwrite=args.overwrite,
    )

    summary = build_review_summary(
        records
    )

    print(
        "Adaptation review completed."
    )

    print(
        "total:",
        summary["total"],
    )

    print(
        "approved:",
        summary["approved"],
    )

    print(
        "rejected:",
        summary["rejected"],
    )

    print(
        "pending:",
        summary["pending"],
    )


if __name__ == "__main__":
    main()
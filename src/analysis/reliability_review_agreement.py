"""Reliability supplement için iki bağımsız reviewer agreement analizi.

Bu modül iki reviewer CSV dosyasını karşılaştırır ve her review alanı için:

- raw agreement,
- Cohen's kappa,
- label distributions,
- missing-label counts,
- disagreement records

üretir.

Reviewer dosyalarını değiştirmez.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REVIEW_FIELDS = (
    "is_unanswerable",
    "semantic_equivalent_en_az",
    "natural_azerbaijani",
    "category_valid",
    "difficulty_valid",
)

VALID_LABELS = {
    "yes",
    "no",
    "uncertain",
}


def normalize_label(value: str | None) -> str:
    """Reviewer label'ını karşılaştırma için normalize eder."""

    if value is None:
        return ""

    return value.strip().lower()


def load_review_csv(
    path: Path,
) -> dict[str, dict[str, str]]:
    """Reviewer CSV dosyasını pair_id tabanlı sözlük olarak yükler."""

    if not path.exists():
        raise FileNotFoundError(
            f"Review file not found: {path}"
        )

    records: dict[str, dict[str, str]] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"Review file has no header: {path}"
            )

        required_fields = {
            "pair_id",
            *REVIEW_FIELDS,
        }

        missing_columns = (
            required_fields
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            pair_id = normalize_label(
                row.get("pair_id")
            )

            if not pair_id:
                raise ValueError(
                    f"{path}: missing pair_id "
                    f"at line {line_number}"
                )

            if pair_id in records:
                raise ValueError(
                    f"{path}: duplicate pair_id "
                    f"{pair_id!r}"
                )

            normalized_row = {
                key: normalize_label(value)
                for key, value in row.items()
                if key is not None
            }

            records[pair_id] = normalized_row

    return records


def validate_labels(
    records: dict[str, dict[str, str]],
    path: Path,
) -> None:
    """Review alanlarındaki label değerlerini doğrular."""

    for pair_id, row in records.items():
        for field in REVIEW_FIELDS:
            value = row.get(field, "")

            if not value:
                continue

            if value not in VALID_LABELS:
                raise ValueError(
                    f"{path}: invalid label "
                    f"{value!r} for field "
                    f"{field!r} in pair "
                    f"{pair_id!r}"
                )


def validate_pair_sets(
    reviewer1: dict[str, dict[str, str]],
    reviewer2: dict[str, dict[str, str]],
) -> None:
    """İki reviewer'ın aynı pair_id setini içerdiğini doğrular."""

    ids1 = set(reviewer1)
    ids2 = set(reviewer2)

    if ids1 == ids2:
        return

    only_reviewer1 = sorted(
        ids1 - ids2
    )
    only_reviewer2 = sorted(
        ids2 - ids1
    )

    raise ValueError(
        "Reviewer files do not contain identical pair_id sets. "
        f"Only reviewer1: {only_reviewer1}; "
        f"only reviewer2: {only_reviewer2}"
    )


def cohen_kappa(
    labels1: list[str],
    labels2: list[str],
) -> float | None:
    """Nominal Cohen's kappa hesaplar.

    Missing label içeren pair'ler bu fonksiyona verilmemelidir.

    Beklenen agreement 1.0 ise kappa matematiksel olarak tanımsızdır
    ve None döndürülür.
    """

    if len(labels1) != len(labels2):
        raise ValueError(
            "Label vectors must have equal length."
        )

    total = len(labels1)

    if total == 0:
        return None

    observed_agreement = sum(
        label1 == label2
        for label1, label2 in zip(
            labels1,
            labels2,
        )
    ) / total

    counts1 = Counter(labels1)
    counts2 = Counter(labels2)

    categories = (
        set(counts1)
        | set(counts2)
    )

    expected_agreement = sum(
        (
            counts1[category] / total
        )
        * (
            counts2[category] / total
        )
        for category in categories
    )

    denominator = (
        1.0 - expected_agreement
    )

    if denominator == 0.0:
        return None

    return (
        observed_agreement
        - expected_agreement
    ) / denominator


def analyze_field(
    reviewer1: dict[str, dict[str, str]],
    reviewer2: dict[str, dict[str, str]],
    field: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Tek review alanının agreement analizini yapar."""

    complete_pairs: list[
        tuple[str, str, str]
    ] = []

    missing_pairs: list[str] = []

    for pair_id in sorted(reviewer1):
        label1 = reviewer1[
            pair_id
        ].get(
            field,
            "",
        )

        label2 = reviewer2[
            pair_id
        ].get(
            field,
            "",
        )

        if not label1 or not label2:
            missing_pairs.append(
                pair_id
            )
            continue

        complete_pairs.append(
            (
                pair_id,
                label1,
                label2,
            )
        )

    labels1 = [
        label1
        for _, label1, _ in complete_pairs
    ]

    labels2 = [
        label2
        for _, _, label2 in complete_pairs
    ]

    total_complete = len(
        complete_pairs
    )

    agreement_count = sum(
        label1 == label2
        for label1, label2 in zip(
            labels1,
            labels2,
        )
    )

    raw_agreement = (
        agreement_count / total_complete
        if total_complete
        else None
    )

    disagreements = [
        {
            "pair_id": pair_id,
            "field": field,
            "reviewer1": label1,
            "reviewer2": label2,
        }
        for (
            pair_id,
            label1,
            label2,
        ) in complete_pairs
        if label1 != label2
    ]

    summary = {
        "field": field,
        "complete_pairs": total_complete,
        "missing_pairs": len(
            missing_pairs
        ),
        "agreements": agreement_count,
        "disagreements": len(
            disagreements
        ),
        "raw_agreement": raw_agreement,
        "cohen_kappa": cohen_kappa(
            labels1,
            labels2,
        ),
        "reviewer1_distribution": dict(
            sorted(
                Counter(
                    labels1
                ).items()
            )
        ),
        "reviewer2_distribution": dict(
            sorted(
                Counter(
                    labels2
                ).items()
            )
        ),
        "missing_pair_ids": (
            missing_pairs
        ),
    }

    return (
        summary,
        disagreements,
    )


def save_json(
    data: Any,
    path: Path,
) -> None:
    """JSON artifact kaydeder."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def save_summary_csv(
    summaries: list[dict[str, Any]],
    path: Path,
) -> None:
    """Field-level agreement summary CSV artifact'ını kaydeder."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "field",
        "complete_pairs",
        "missing_pairs",
        "agreements",
        "disagreements",
        "raw_agreement",
        "cohen_kappa",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()

        for summary in summaries:
            writer.writerow({
                key: summary[key]
                for key in fieldnames
            })


def save_disagreements_csv(
    disagreements: list[
        dict[str, str]
    ],
    path: Path,
) -> None:
    """Disagreement kayıtlarını CSV olarak kaydeder."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "pair_id",
        "field",
        "reviewer1",
        "reviewer2",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            disagreements
        )


def analyze_reviews(
    reviewer1_path: Path,
    reviewer2_path: Path,
    expected_pairs: int | None = 60,
) -> dict[str, Any]:
    """İki reviewer dosyasının tüm agreement analizini çalıştırır."""

    reviewer1 = load_review_csv(
        reviewer1_path
    )

    reviewer2 = load_review_csv(
        reviewer2_path
    )

    validate_labels(
        reviewer1,
        reviewer1_path,
    )

    validate_labels(
        reviewer2,
        reviewer2_path,
    )

    validate_pair_sets(
        reviewer1,
        reviewer2,
    )

    if (
        expected_pairs is not None
        and len(reviewer1)
        != expected_pairs
    ):
        raise ValueError(
            f"Expected {expected_pairs} review pairs, "
            f"found {len(reviewer1)}."
        )

    summaries: list[
        dict[str, Any]
    ] = []

    all_disagreements: list[
        dict[str, str]
    ] = []

    for field in REVIEW_FIELDS:
        (
            summary,
            disagreements,
        ) = analyze_field(
            reviewer1,
            reviewer2,
            field,
        )

        summaries.append(
            summary
        )

        all_disagreements.extend(
            disagreements
        )

    return {
        "pairs": len(
            reviewer1
        ),
        "review_fields": list(
            REVIEW_FIELDS
        ),
        "summaries": summaries,
        "disagreements": (
            all_disagreements
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """CLI argümanlarını parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute inter-reviewer agreement "
            "for the reliability supplement."
        )
    )

    parser.add_argument(
        "--reviewer1",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reviewer2",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=60,
    )

    return parser.parse_args()


def main() -> None:
    """IAA analysis CLI entry point."""

    args = parse_arguments()

    results = analyze_reviews(
        reviewer1_path=(
            args.reviewer1
        ),
        reviewer2_path=(
            args.reviewer2
        ),
        expected_pairs=(
            args.expected_pairs
        ),
    )

    summaries = results[
        "summaries"
    ]

    disagreements = results[
        "disagreements"
    ]

    save_json(
        results,
        args.output_dir
        / "agreement_summary.json",
    )

    save_summary_csv(
        summaries,
        args.output_dir
        / "agreement_summary.csv",
    )

    save_disagreements_csv(
        disagreements,
        args.output_dir
        / "disagreements.csv",
    )

    print(
        f"IAA analysis completed for "
        f"{results['pairs']} pairs."
    )

    for summary in summaries:
        print(
            summary["field"],
            "| complete=",
            summary["complete_pairs"],
            "| agreement=",
            summary["raw_agreement"],
            "| kappa=",
            summary["cohen_kappa"],
        )

    print(
        "Disagreement records:",
        len(disagreements),
    )


if __name__ == "__main__":
    main()

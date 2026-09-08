"""Adaptation datası ile frozen benchmark arasında overlap kontrolü yapar.

Bu modül adaptation instruction'larını benchmark question'larıyla
karşılaştırır ve doğrudan ya da yüksek token-overlap taşıyan
eşleşmeleri raporlar.

Amaç training leakage riskini training başlamadan önce tespit etmektir.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(
    r"[0-9A-Za-zƏəĞğİıÖöŞşÜüÇç]+"
)


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """JSONL kayıtlarını yükler."""

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
                    f"Invalid JSON at line {line_number}: {path}"
                ) from error

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} must be a JSON object."
                )

            rows.append(row)

    return rows


def normalize_text(
    text: str,
) -> str:
    """Metni overlap karşılaştırması için normalize eder."""

    tokens = TOKEN_PATTERN.findall(
        text.lower()
    )

    return " ".join(tokens)


def token_set(
    text: str,
) -> set[str]:
    """Normalize edilmiş token kümesini döndürür."""

    return set(
        normalize_text(text).split()
    )


def jaccard_similarity(
    first: str,
    second: str,
) -> float:
    """İki metin arasında token-set Jaccard similarity hesaplar."""

    first_tokens = token_set(
        first
    )

    second_tokens = token_set(
        second
    )

    if not first_tokens and not second_tokens:
        return 1.0

    if not first_tokens or not second_tokens:
        return 0.0

    intersection = (
        first_tokens
        & second_tokens
    )

    union = (
        first_tokens
        | second_tokens
    )

    return len(intersection) / len(union)


def get_adaptation_text(
    row: dict[str, Any],
) -> str:
    """Adaptation kaydından karşılaştırılacak instruction metnini alır."""

    instruction = row.get(
        "instruction"
    )

    if not isinstance(
        instruction,
        str,
    ):
        raise ValueError(
            f"Invalid adaptation instruction: "
            f"{row.get('item_id', '<unknown>')}"
        )

    return instruction


def get_benchmark_text(
    row: dict[str, Any],
) -> str:
    """Benchmark kaydından karşılaştırılacak question metnini alır."""

    question = row.get(
        "question"
    )

    if not isinstance(
        question,
        str,
    ):
        raise ValueError(
            f"Invalid benchmark question: "
            f"{row.get('item_id', '<unknown>')}"
        )

    return question


def find_exact_overlaps(
    adaptation_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize edilmiş exact overlap kayıtlarını bulur."""

    benchmark_index: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for benchmark_row in benchmark_rows:
        text = normalize_text(
            get_benchmark_text(
                benchmark_row
            )
        )

        benchmark_index.setdefault(
            text,
            [],
        ).append(
            benchmark_row
        )

    matches: list[dict[str, Any]] = []

    for adaptation_row in adaptation_rows:
        text = normalize_text(
            get_adaptation_text(
                adaptation_row
            )
        )

        for benchmark_row in benchmark_index.get(
            text,
            [],
        ):
            matches.append(
                {
                    "adaptation_item_id": (
                        adaptation_row["item_id"]
                    ),
                    "benchmark_item_id": (
                        benchmark_row["item_id"]
                    ),
                    "match_type": "exact_normalized",
                    "similarity": 1.0,
                    "adaptation_text": (
                        get_adaptation_text(
                            adaptation_row
                        )
                    ),
                    "benchmark_text": (
                        get_benchmark_text(
                            benchmark_row
                        )
                    ),
                }
            )

    return matches


def find_high_similarity_overlaps(
    adaptation_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Threshold üzerindeki token-overlap adaylarını bulur."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0.0 and 1.0."
        )

    matches: list[dict[str, Any]] = []

    for adaptation_row in adaptation_rows:
        adaptation_text = get_adaptation_text(
            adaptation_row
        )

        normalized_adaptation = normalize_text(
            adaptation_text
        )

        for benchmark_row in benchmark_rows:
            benchmark_text = get_benchmark_text(
                benchmark_row
            )

            normalized_benchmark = normalize_text(
                benchmark_text
            )

            if (
                normalized_adaptation
                == normalized_benchmark
            ):
                continue

            similarity = jaccard_similarity(
                adaptation_text,
                benchmark_text,
            )

            if similarity >= threshold:
                matches.append(
                    {
                        "adaptation_item_id": (
                            adaptation_row["item_id"]
                        ),
                        "benchmark_item_id": (
                            benchmark_row["item_id"]
                        ),
                        "match_type": "high_token_overlap",
                        "similarity": similarity,
                        "adaptation_text": adaptation_text,
                        "benchmark_text": benchmark_text,
                    }
                )

    return matches


def build_overlap_report(
    adaptation_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Tam overlap audit raporunu oluşturur."""

    exact_matches = find_exact_overlaps(
        adaptation_rows=adaptation_rows,
        benchmark_rows=benchmark_rows,
    )

    high_similarity_matches = (
        find_high_similarity_overlaps(
            adaptation_rows=adaptation_rows,
            benchmark_rows=benchmark_rows,
            threshold=threshold,
        )
    )

    return {
        "adaptation_records": len(
            adaptation_rows
        ),
        "benchmark_records": len(
            benchmark_rows
        ),
        "threshold": threshold,
        "exact_overlap_count": len(
            exact_matches
        ),
        "high_similarity_count": len(
            high_similarity_matches
        ),
        "exact_overlaps": exact_matches,
        "high_similarity_overlaps": (
            high_similarity_matches
        ),
    }


def save_report(
    report: dict[str, Any],
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """Overlap raporunu JSON olarak kaydeder."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """CLI argumentlerini parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Check adaptation data for benchmark overlap."
        )
    )

    parser.add_argument(
        "--adaptation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_arguments()

    adaptation_rows = load_jsonl(
        args.adaptation
    )

    benchmark_rows = load_jsonl(
        args.benchmark
    )

    report = build_overlap_report(
        adaptation_rows=adaptation_rows,
        benchmark_rows=benchmark_rows,
        threshold=args.threshold,
    )

    save_report(
        report=report,
        output_path=args.output,
        overwrite=args.overwrite,
    )

    print(
        "Adaptation overlap audit completed."
    )
    print(
        "exact overlaps:",
        report["exact_overlap_count"],
    )
    print(
        "high similarity:",
        report["high_similarity_count"],
    )


if __name__ == "__main__":
    main()
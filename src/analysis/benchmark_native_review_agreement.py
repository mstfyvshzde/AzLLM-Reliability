from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

FIELDS = (
    "semantic_equivalent_en_az",
    "natural_azerbaijani",
    "reference_answer_equivalent",
    "answerability_equivalent",
)

VALID = {"yes", "no", "uncertain"}


def load_reviews(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}

    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pair_id = row["pair_id"].strip()

            if pair_id in rows:
                raise ValueError(f"Duplicate pair_id: {pair_id}")

            rows[pair_id] = row

    return rows


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b):
        raise ValueError("Label vectors differ in length.")

    n = len(a)
    if n == 0:
        raise ValueError("No labels available.")

    observed = sum(x == y for x, y in zip(a, b)) / n

    ca = Counter(a)
    cb = Counter(b)

    expected = sum(
        (ca[label] / n) * (cb[label] / n)
        for label in VALID
    )

    if expected == 1.0:
        return 1.0

    return (observed - expected) / (1.0 - expected)


def main() -> None:
    r1_path = Path(
        "data/interim/benchmark_native_review/"
        "reviewer1/benchmark_review.csv"
    )
    r2_path = Path(
        "data/interim/benchmark_native_review/"
        "reviewer2/benchmark_review.csv"
    )

    r1 = load_reviews(r1_path)
    r2 = load_reviews(r2_path)

    if set(r1) != set(r2):
        raise ValueError("Reviewer pair_id sets differ.")

    print(f"Pairs: {len(r1)}")

    for field in FIELDS:
        a = []
        b = []

        for pair_id in sorted(r1):
            x = r1[pair_id][field].strip().lower()
            y = r2[pair_id][field].strip().lower()

            if x not in VALID or y not in VALID:
                raise ValueError(
                    f"{pair_id}: incomplete/invalid label for {field}"
                )

            a.append(x)
            b.append(y)

        agreement = sum(x == y for x, y in zip(a, b)) / len(a)
        kappa = cohen_kappa(a, b)

        print()
        print(field)
        print(f"  raw agreement: {agreement:.2%}")
        print(f"  Cohen's kappa: {kappa:.3f}")


if __name__ == "__main__":
    main()

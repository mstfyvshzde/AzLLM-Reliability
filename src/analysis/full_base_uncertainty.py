from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_pairs(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("metadata", {}).get("is_answerable") is True:
                    rows.append(row)
    return rows


def metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows)

    en = sum(r["source_correct"] for r in rows) / n
    az = sum(r["target_correct"] for r in rows) / n
    gap = en - az

    return {
        "en_accuracy": en,
        "az_accuracy": az,
        "en_az_gap": gap,
    }


def percentile(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    values = sorted(values)
    lo = int((alpha / 2) * len(values))
    hi = int((1 - alpha / 2) * len(values)) - 1
    return values[lo], values[hi]


def bootstrap(rows: list[dict], samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    observed = metrics(rows)

    distributions = {
        "en_accuracy": [],
        "az_accuracy": [],
        "en_az_gap": [],
    }

    for _ in range(samples):
        sample = [
            rng.choice(rows)
            for _ in range(len(rows))
        ]

        m = metrics(sample)

        for key in distributions:
            distributions[key].append(m[key])

    ci = {}

    for key, values in distributions.items():
        lo, hi = percentile(values)
        ci[key] = {
            "lower": lo,
            "upper": hi,
        }

    return {
        "n_pairs": len(rows),
        "samples": samples,
        "seed": seed,
        "observed": observed,
        "ci_95": ci,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rows = load_pairs(args.input)

    if not rows:
        raise ValueError("No answerable pairs found.")

    result = bootstrap(
        rows,
        samples=args.samples,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

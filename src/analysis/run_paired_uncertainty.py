"""Final base-vs-adapted paired uncertainty analysis runner."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.analysis.paired_uncertainty import (
    PairedComparison,
    paired_bootstrap,
)


BASE_PATH = Path(
    "data/results/final_baseline/"
    "test_eval_base/task_aware_results.jsonl"
)

ADAPTED_PATH = Path(
    "data/results/adaptation_pilot_v2.0/"
    "test_eval_0090_readjudicated/task_aware_results.jsonl"
)

OUTPUT_PATH = Path(
    "data/results/adaptation_pilot_v2.0/"
    "final_stats/paired_uncertainty_base_vs_adapted_0090.json"
)


def load_answerable(path: Path) -> dict[str, dict]:
    """Answerable task-aware results keyed by item_id."""

    rows: dict[str, dict] = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if not row.get(
                "metadata",
                {},
            ).get(
                "is_answerable",
                False,
            ):
                continue

            rows[row["item_id"]] = row

    return rows


def build_paired_comparisons() -> list[PairedComparison]:
    """Build complete EN-AZ semantic-pair comparisons."""

    base = load_answerable(BASE_PATH)
    adapted = load_answerable(ADAPTED_PATH)

    if set(base) != set(adapted):
        raise ValueError(
            "Base and adapted answerable item sets differ."
        )

    grouped: dict[
        str,
        dict[str, dict[str, int]],
    ] = defaultdict(dict)

    for item_id, base_row in base.items():
        adapted_row = adapted[item_id]

        if (
            base_row["pair_id"]
            != adapted_row["pair_id"]
        ):
            raise ValueError(
                f"pair_id mismatch for {item_id}"
            )

        if (
            base_row["language"]
            != adapted_row["language"]
        ):
            raise ValueError(
                f"language mismatch for {item_id}"
            )

        grouped[
            base_row["pair_id"]
        ][
            base_row["language"]
        ] = {
            "base": int(
                base_row["correct"]
            ),
            "adapted": int(
                adapted_row["correct"]
            ),
        }

    comparisons: list[PairedComparison] = []

    for pair_id in sorted(grouped):
        pair = grouped[pair_id]

        if set(pair) != {"en", "az"}:
            raise ValueError(
                f"Incomplete EN-AZ pair: {pair_id}"
            )

        comparisons.append(
            PairedComparison(
                pair_id=pair_id,
                base_en=pair["en"]["base"],
                base_az=pair["az"]["base"],
                adapted_en=pair["en"]["adapted"],
                adapted_az=pair["az"]["adapted"],
            )
        )

    return comparisons


def main() -> None:
    comparisons = build_paired_comparisons()

    result = paired_bootstrap(
        comparisons,
        samples=10_000,
        seed=17,
    )

    result["correct_counts"] = {
        "base_en": sum(
            pair.base_en
            for pair in comparisons
        ),
        "base_az": sum(
            pair.base_az
            for pair in comparisons
        ),
        "adapted_en": sum(
            pair.adapted_en
            for pair in comparisons
        ),
        "adapted_az": sum(
            pair.adapted_az
            for pair in comparisons
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()

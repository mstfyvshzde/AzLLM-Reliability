"""Summarize V2 development checkpoint performance."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    "data/results/adaptation_pilot_v2.0"
)

CHECKPOINTS = [
    "0090",
    "0180",
    "0270",
    "0360",
    "0450",
]

OUTPUT_PATH = (
    ROOT
    / "final_stats"
    / "dev_checkpoint_summary_v2.0.json"
)


def main() -> None:
    rows = []

    for checkpoint in CHECKPOINTS:
        path = (
            ROOT
            / f"dev_eval_{checkpoint}_readjudicated"
            / "primary_capability_summary.json"
        )

        summary = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        rows.append(
            {
                "checkpoint": checkpoint,
                "overall_accuracy": summary[
                    "overall"
                ]["accuracy"],
                "az_accuracy": summary[
                    "by_language"
                ]["az"]["accuracy"],
                "en_accuracy": summary[
                    "by_language"
                ]["en"]["accuracy"],
                "language_gap": summary[
                    "language_gap"
                ]["absolute_gap"],
                "overall_correct": summary[
                    "overall"
                ]["correct"],
                "overall_total": summary[
                    "overall"
                ]["total"],
            }
        )

    output = {
        "selection_split": "dev",
        "selected_checkpoint": "0090",
        "selection_note": (
            "Checkpoint 0090 had the highest observed "
            "overall and Azerbaijani primary capability "
            "and the smallest observed EN-AZ gap on dev. "
            "Because the dev split is small, these values "
            "should be interpreted as checkpoint-selection "
            "signals rather than precise performance estimates."
        ),
        "checkpoints": rows,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

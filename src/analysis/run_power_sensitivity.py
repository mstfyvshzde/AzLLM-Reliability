"""Run paired McNemar power-sensitivity analysis for final comparison."""

from __future__ import annotations

import json
from pathlib import Path

from src.analysis.power_sensitivity import (
    find_minimum_detectable_effect,
)


INPUT_PATH = Path(
    "data/results/adaptation_pilot_v2.0/"
    "final_stats/base_vs_adapted_0090.json"
)

OUTPUT_PATH = Path(
    "data/results/adaptation_pilot_v2.0/"
    "final_stats/power_sensitivity_base_vs_adapted_0090.json"
)


def main() -> None:
    stats = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    cases = {
        "overall": stats["overall"],
        "az": stats["by_language"]["az"],
        "en": stats["by_language"]["en"],
    }

    results = {}

    for name, section in cases.items():
        n = int(section["n"])

        transitions = section["transitions"]

        discordant = (
            int(transitions["base_only_correct"])
            + int(transitions["adapted_only_correct"])
        )

        discordance_rate = discordant / n

        mde = find_minimum_detectable_effect(
            n=n,
            discordance_rate=discordance_rate,
            target_power=0.80,
            alpha=0.05,
        )

        results[name] = {
            "n": n,
            "discordant_pairs": discordant,
            "discordance_rate": discordance_rate,
            "mde_absolute_difference": mde,
            "mde_percentage_points": mde * 100,
            "alpha": 0.05,
            "target_power": 0.80,
        }

    output = {
        "analysis_type": (
            "observed-discordance McNemar "
            "power sensitivity analysis"
        ),
        "interpretation": (
            "Sensitivity analysis based on observed discordance rates; "
            "not a prospective design power analysis."
        ),
        "results": results,
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

    print(
        f"Wrote: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()

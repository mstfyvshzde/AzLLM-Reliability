"""Build Azerbaijani scoring-audit packet for held-out test predictions."""

from __future__ import annotations

import json
from pathlib import Path


SOURCES = {
    "llama_base": Path(
        "data/results/final_baseline/test_eval_base/task_aware_results.jsonl"
    ),
    "adapted_v2_0090": Path(
        "data/results/adaptation_pilot_v2.0/"
        "test_eval_0090_readjudicated/task_aware_results.jsonl"
    ),
    "qwen_base": Path(
        "data/results/qwen2_5_7b_baseline/"
        "test_eval_readjudicated/task_aware_results.jsonl"
    ),
}

BENCHMARK_PATH = Path(
    "data/processed/final/benchmark_v1.0_test.jsonl"
)

OUTPUT = Path(
    "data/interim/az_scoring_audit/"
    "az_test_scoring_audit_v1.0.jsonl"
)


def main() -> None:
    benchmark: dict[str, dict] = {}

    with BENCHMARK_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            benchmark[row["item_id"]] = row

    grouped: dict[str, dict] = {}

    for model_name, path in SOURCES.items():
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                if row["language"] != "az":
                    continue

                if not row.get("metadata", {}).get(
                    "is_answerable", False
                ):
                    continue

                item = grouped.setdefault(
                    row["item_id"],
                    {
                        "item_id": row["item_id"],
                        "pair_id": row["pair_id"],
                        "task": row["task"],
                        "question": benchmark[row["item_id"]]["question"],
                        "reference_answer": row["reference_answer"],
                        "predictions": {},
                        "audit": {
                            "reference_valid": "",
                            "llama_base_correct": "",
                            "adapted_v2_0090_correct": "",
                            "qwen_base_correct": "",
                            "scoring_issue": "",
                            "reviewer_notes": "",
                        },
                    },
                )

                item["predictions"][model_name] = {
                    "prediction": row["prediction"],
                    "automated_correct": int(row["correct"]),
                    "evaluator": row["evaluator"],
                }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", encoding="utf-8") as f:
        for item_id in sorted(grouped):
            f.write(
                json.dumps(
                    grouped[item_id],
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote: {OUTPUT}")
    print(f"AZ answerable items: {len(grouped)}")


if __name__ == "__main__":
    main()

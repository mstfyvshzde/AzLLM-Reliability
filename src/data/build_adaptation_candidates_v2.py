"""Build deterministic adaptation-v2 candidate records.

This module converts the frozen v2 generation quotas into a candidate
pool before any synthetic question/answer generation happens.

Important:
    - no benchmark examples are copied here
    - no model generation happens here
    - candidate counts must exactly match generation_v2.0.yaml
    - IDs are deterministic under the configured seed/order
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AdaptationCandidateV2:
    """Represent one planned adaptation-v2 training example."""

    item_id: str
    language: str
    task: str
    source: str
    review_status: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the candidate to a JSON-serializable dictionary."""
        return asdict(self)


def load_generation_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate the adaptation-v2 generation config."""

    config_path = Path(path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Generation config must contain a YAML mapping."
        )

    required_keys = {
        "version",
        "seed",
        "total_records",
        "language_quotas",
        "quotas",
    }

    missing = required_keys - config.keys()

    if missing:
        raise ValueError(
            f"Generation config missing keys: {sorted(missing)}"
        )

    return config


def validate_quotas(
    config: dict[str, Any],
) -> None:
    """Validate task and language quotas against the configured total."""

    quotas = config["quotas"]
    language_quotas = config["language_quotas"]

    calculated_total = 0

    for language, task_quotas in quotas.items():
        if language not in language_quotas:
            raise ValueError(
                f"Quota language not declared: {language}"
            )

        language_total = sum(task_quotas.values())

        expected_language_total = language_quotas[language]

        if language_total != expected_language_total:
            raise ValueError(
                f"{language} quota mismatch: "
                f"expected {expected_language_total}, "
                f"got {language_total}"
            )

        calculated_total += language_total

    expected_total = config["total_records"]

    if calculated_total != expected_total:
        raise ValueError(
            f"Total quota mismatch: "
            f"expected {expected_total}, "
            f"got {calculated_total}"
        )


def build_candidates(
    config: dict[str, Any],
) -> list[AdaptationCandidateV2]:
    """Build deterministic candidates from the configured quotas."""

    validate_quotas(config)

    candidates: list[AdaptationCandidateV2] = []

    index = 1

    for language, task_quotas in config["quotas"].items():
        for task, count in task_quotas.items():
            for task_index in range(1, count + 1):
                item_id = f"adapt_v2_{index:04d}"

                candidate = AdaptationCandidateV2(
                    item_id=item_id,
                    language=language,
                    task=task,
                    source="synthetic",
                    review_status="pending",
                    metadata={
                        "version": config["version"],
                        "seed": config["seed"],
                        "task_index": task_index,
                        "generation_status": "pending",
                    },
                )

                candidates.append(candidate)

                index += 1

    expected_total = config["total_records"]

    if len(candidates) != expected_total:
        raise RuntimeError(
            f"Candidate generation produced "
            f"{len(candidates)} records; "
            f"expected {expected_total}."
        )

    item_ids = {
        candidate.item_id
        for candidate in candidates
    }

    if len(item_ids) != len(candidates):
        raise RuntimeError(
            "Duplicate candidate item_id detected."
        )

    return candidates
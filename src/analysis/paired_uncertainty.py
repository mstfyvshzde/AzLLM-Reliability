"""Paired EN-AZ uncertainty analysis helpers.

Bu modül base ve adapted model sonuçlarını pair seviyesinde karşılaştırır.
Amaç:

- language-specific accuracy,
- overall accuracy,
- EN-AZ language gap,
- adaptation sonrası gap change

için paired bootstrap belirsizlik aralıkları üretmektir.

Bootstrap unit'i semantic pair'dir. Böylece aynı pair'e ait EN ve AZ
kayıtları birlikte yeniden örneklenir ve pairing yapısı korunur.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


@dataclass(frozen=True)
class PairedComparison:
    """Tek semantic pair için base/adapted correctness değerleri."""

    pair_id: str
    base_en: int
    base_az: int
    adapted_en: int
    adapted_az: int

    def __post_init__(self) -> None:
        """Correctness alanlarının binary olduğunu doğrular."""

        for field_name in (
            "base_en",
            "base_az",
            "adapted_en",
            "adapted_az",
        ):
            value = getattr(self, field_name)

            if value not in {0, 1}:
                raise ValueError(
                    f"{field_name} must be 0 or 1."
                )


def compute_metrics(
    pairs: Iterable[PairedComparison],
) -> dict[str, float]:
    """Paired comparison kayıtlarından aggregate metrikleri hesaplar."""

    pair_list = list(pairs)

    if not pair_list:
        raise ValueError(
            "Expected at least one paired comparison."
        )

    n = len(pair_list)

    base_en = sum(
        pair.base_en
        for pair in pair_list
    ) / n

    base_az = sum(
        pair.base_az
        for pair in pair_list
    ) / n

    adapted_en = sum(
        pair.adapted_en
        for pair in pair_list
    ) / n

    adapted_az = sum(
        pair.adapted_az
        for pair in pair_list
    ) / n

    base_overall = (
        sum(pair.base_en for pair in pair_list)
        + sum(pair.base_az for pair in pair_list)
    ) / (2 * n)

    adapted_overall = (
        sum(pair.adapted_en for pair in pair_list)
        + sum(pair.adapted_az for pair in pair_list)
    ) / (2 * n)

    base_gap = base_en - base_az
    adapted_gap = adapted_en - adapted_az

    gap_change = adapted_gap - base_gap

    return {
        "base_en": base_en,
        "base_az": base_az,
        "adapted_en": adapted_en,
        "adapted_az": adapted_az,
        "base_overall": base_overall,
        "adapted_overall": adapted_overall,
        "base_gap": base_gap,
        "adapted_gap": adapted_gap,
        "gap_change": gap_change,
    }


def percentile_interval(
    values: Iterable[float],
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval döndürür."""

    ordered = sorted(values)

    if not ordered:
        raise ValueError(
            "Expected at least one bootstrap value."
        )

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    lower_index = int(
        (alpha / 2) * len(ordered)
    )

    upper_index = int(
        (1 - alpha / 2) * len(ordered)
    ) - 1

    return (
        ordered[lower_index],
        ordered[upper_index],
    )


def paired_bootstrap(
    pairs: Iterable[PairedComparison],
    *,
    samples: int = 10_000,
    seed: int = 17,
) -> dict[str, object]:
    """Semantic pair seviyesinde paired bootstrap uygular."""

    pair_list = list(pairs)

    if not pair_list:
        raise ValueError(
            "Expected at least one paired comparison."
        )

    if samples <= 0:
        raise ValueError(
            "samples must be greater than zero."
        )

    observed = compute_metrics(
        pair_list
    )

    rng = random.Random(seed)

    distributions: dict[str, list[float]] = {
        key: []
        for key in observed
    }

    for _ in range(samples):
        sample = [
            rng.choice(pair_list)
            for _ in range(len(pair_list))
        ]

        metrics = compute_metrics(
            sample
        )

        for key, value in metrics.items():
            distributions[key].append(
                value
            )

    ci_95 = {
        key: {
            "lower": percentile_interval(
                values,
                alpha=0.05,
            )[0],
            "upper": percentile_interval(
                values,
                alpha=0.05,
            )[1],
        }
        for key, values in distributions.items()
    }

    return {
        "seed": seed,
        "samples": samples,
        "n_pairs": len(pair_list),
        "observed": observed,
        "ci_95": ci_95,
    }

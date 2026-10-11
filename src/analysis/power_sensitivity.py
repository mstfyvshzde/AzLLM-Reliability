"""Paired McNemar power-sensitivity helpers.

Bu modül paired binary outcome'lar için observed discordance rate'e dayalı
yaklaşık power ve minimum detectable effect (MDE) hesabı yapar.

Not:
Bu analiz prospective design power analysis değil, sensitivity analysis'tir.
Observed discordance rate sabit kabul edilerek belirli bir absolute paired
difference'ın exact McNemar test ile yakalanma olasılığı hesaplanır.
"""

from __future__ import annotations

import math


def _binomial_pmf(
    k: int,
    n: int,
    p: float,
) -> float:
    """Binomial probability mass function."""

    return (
        math.comb(n, k)
        * (p ** k)
        * ((1 - p) ** (n - k))
    )


def _exact_two_sided_binomial_p(
    k: int,
    n: int,
) -> float:
    """Exact two-sided binomial p-value under p=0.5."""

    probabilities = [
        _binomial_pmf(
            i,
            n,
            0.5,
        )
        for i in range(n + 1)
    ]

    observed_probability = probabilities[k]

    return min(
        1.0,
        sum(
            probability
            for probability in probabilities
            if probability
            <= observed_probability + 1e-15
        ),
    )


def exact_mcnemar_power(
    *,
    n: int,
    discordance_rate: float,
    absolute_difference: float,
    alpha: float = 0.05,
) -> float:
    """Exact McNemar rejection probability for a paired effect size."""

    if n <= 0:
        raise ValueError(
            "n must be greater than zero."
        )

    if not 0 < discordance_rate <= 1:
        raise ValueError(
            "discordance_rate must be between 0 and 1."
        )

    if not 0 <= absolute_difference <= discordance_rate:
        raise ValueError(
            "absolute_difference must be between 0 and discordance_rate."
        )

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    theta = (
        discordance_rate
        + absolute_difference
    ) / (
        2 * discordance_rate
    )

    power = 0.0

    for discordant_n in range(n + 1):
        probability_discordant_n = _binomial_pmf(
            discordant_n,
            n,
            discordance_rate,
        )

        rejection_probability = 0.0

        for adapted_only in range(
            discordant_n + 1
        ):
            p_value = _exact_two_sided_binomial_p(
                adapted_only,
                discordant_n,
            )

            if p_value <= alpha:
                rejection_probability += _binomial_pmf(
                    adapted_only,
                    discordant_n,
                    theta,
                )

        power += (
            probability_discordant_n
            * rejection_probability
        )

    return power


def find_minimum_detectable_effect(
    *,
    n: int,
    discordance_rate: float,
    target_power: float = 0.80,
    alpha: float = 0.05,
) -> float:
    """Target power'a ulaşan minimum absolute paired difference'ı bulur."""

    if not 0 < target_power < 1:
        raise ValueError(
            "target_power must be between 0 and 1."
        )

    maximum_power = exact_mcnemar_power(
        n=n,
        discordance_rate=discordance_rate,
        absolute_difference=discordance_rate,
        alpha=alpha,
    )

    if maximum_power < target_power:
        raise ValueError(
            "Target power is unattainable for the supplied parameters."
        )

    low = 0.0
    high = discordance_rate

    for _ in range(60):
        midpoint = (
            low + high
        ) / 2

        power = exact_mcnemar_power(
            n=n,
            discordance_rate=discordance_rate,
            absolute_difference=midpoint,
            alpha=alpha,
        )

        if power >= target_power:
            high = midpoint
        else:
            low = midpoint

    return high

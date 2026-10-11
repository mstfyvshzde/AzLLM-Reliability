"""Paired McNemar power-sensitivity yardımcılarını test eder."""

import pytest

from src.analysis.power_sensitivity import (
    exact_mcnemar_power,
    find_minimum_detectable_effect,
)


def test_power_increases_with_larger_effect() -> None:
    small = exact_mcnemar_power(
        n=61,
        discordance_rate=15 / 61,
        absolute_difference=0.05,
    )

    large = exact_mcnemar_power(
        n=61,
        discordance_rate=15 / 61,
        absolute_difference=0.15,
    )

    assert large > small


def test_find_mde_returns_expected_range() -> None:
    mde = find_minimum_detectable_effect(
        n=61,
        discordance_rate=15 / 61,
        target_power=0.80,
    )

    assert mde == pytest.approx(
        0.1797,
        abs=0.002,
    )


def test_invalid_difference_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and discordance_rate",
    ):
        exact_mcnemar_power(
            n=61,
            discordance_rate=0.20,
            absolute_difference=0.30,
        )


def test_invalid_target_power_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="target_power",
    ):
        find_minimum_detectable_effect(
            n=61,
            discordance_rate=0.20,
            target_power=1.2,
        )

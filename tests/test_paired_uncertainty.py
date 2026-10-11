"""Paired EN-AZ uncertainty analysis yardımcılarını test eder."""

import pytest

from src.analysis.paired_uncertainty import (
    PairedComparison,
    compute_metrics,
    paired_bootstrap,
    percentile_interval,
)


def make_pair(
    *,
    pair_id: str,
    base_en: int,
    base_az: int,
    adapted_en: int,
    adapted_az: int,
) -> PairedComparison:
    """Testler için küçük bir paired comparison oluşturur."""

    return PairedComparison(
        pair_id=pair_id,
        base_en=base_en,
        base_az=base_az,
        adapted_en=adapted_en,
        adapted_az=adapted_az,
    )


def test_compute_metrics() -> None:
    pairs = [
        make_pair(
            pair_id="pair_1",
            base_en=1,
            base_az=0,
            adapted_en=1,
            adapted_az=1,
        ),
        make_pair(
            pair_id="pair_2",
            base_en=1,
            base_az=1,
            adapted_en=0,
            adapted_az=1,
        ),
    ]

    metrics = compute_metrics(pairs)

    assert metrics["base_en"] == pytest.approx(1.0)
    assert metrics["base_az"] == pytest.approx(0.5)
    assert metrics["adapted_en"] == pytest.approx(0.5)
    assert metrics["adapted_az"] == pytest.approx(1.0)

    assert metrics["base_overall"] == pytest.approx(0.75)
    assert metrics["adapted_overall"] == pytest.approx(0.75)

    assert metrics["base_gap"] == pytest.approx(0.5)
    assert metrics["adapted_gap"] == pytest.approx(-0.5)
    assert metrics["gap_change"] == pytest.approx(-1.0)


def test_percentile_interval() -> None:
    values = list(range(100))

    lower, upper = percentile_interval(values)

    assert lower == 2
    assert upper == 96


def test_paired_bootstrap_is_reproducible() -> None:
    pairs = [
        make_pair(
            pair_id="pair_1",
            base_en=1,
            base_az=0,
            adapted_en=1,
            adapted_az=1,
        ),
        make_pair(
            pair_id="pair_2",
            base_en=1,
            base_az=1,
            adapted_en=0,
            adapted_az=1,
        ),
        make_pair(
            pair_id="pair_3",
            base_en=0,
            base_az=0,
            adapted_en=1,
            adapted_az=0,
        ),
    ]

    first = paired_bootstrap(
        pairs,
        samples=1000,
        seed=17,
    )

    second = paired_bootstrap(
        pairs,
        samples=1000,
        seed=17,
    )

    assert first == second
    assert first["seed"] == 17
    assert first["samples"] == 1000
    assert "observed" in first
    assert "ci_95" in first
    assert "gap_change" in first["ci_95"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_en", -1),
        ("base_az", 2),
        ("adapted_en", 3),
        ("adapted_az", -2),
    ],
)
def test_paired_comparison_rejects_non_binary_scores(
    field: str,
    value: int,
) -> None:
    kwargs = {
        "pair_id": "pair_1",
        "base_en": 1,
        "base_az": 1,
        "adapted_en": 1,
        "adapted_az": 1,
    }
    kwargs[field] = value

    with pytest.raises(
        ValueError,
        match="must be 0 or 1",
    ):
        PairedComparison(**kwargs)


def test_compute_metrics_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="at least one paired comparison",
    ):
        compute_metrics([])

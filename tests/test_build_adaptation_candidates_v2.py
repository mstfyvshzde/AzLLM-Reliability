"""Tests for the adaptation-v2 candidate builder."""

from pathlib import Path

import pytest

from src.data.build_adaptation_candidates_v2 import (
    build_candidates,
    load_generation_config,
    validate_quotas,
)


CONFIG_PATH = Path(
    "configs/adaptation/generation_v2.0.yaml"
)


def test_load_generation_config() -> None:
    """Generation config should load as a mapping."""

    config = load_generation_config(
        CONFIG_PATH
    )

    assert isinstance(config, dict)
    assert config["version"] == "2.0"
    assert config["total_records"] == 1000


def test_validate_quotas() -> None:
    """Configured language/task quotas should be internally consistent."""

    config = load_generation_config(
        CONFIG_PATH
    )

    validate_quotas(config)


def test_build_candidates_count() -> None:
    """Candidate builder should create exactly 1000 records."""

    config = load_generation_config(
        CONFIG_PATH
    )

    candidates = build_candidates(
        config
    )

    assert len(candidates) == 1000


def test_candidate_ids_are_unique() -> None:
    """Every adaptation-v2 candidate should have a unique item_id."""

    config = load_generation_config(
        CONFIG_PATH
    )

    candidates = build_candidates(
        config
    )

    item_ids = [
        candidate.item_id
        for candidate in candidates
    ]

    assert len(item_ids) == len(set(item_ids))


def test_candidate_id_boundaries() -> None:
    """Candidate IDs should follow the deterministic v2 numbering."""

    config = load_generation_config(
        CONFIG_PATH
    )

    candidates = build_candidates(
        config
    )

    assert candidates[0].item_id == "adapt_v2_0001"
    assert candidates[-1].item_id == "adapt_v2_1000"


def test_language_counts() -> None:
    """Candidate language counts should match the frozen quotas."""

    config = load_generation_config(
        CONFIG_PATH
    )

    candidates = build_candidates(
        config
    )

    az_count = sum(
        candidate.language == "az"
        for candidate in candidates
    )

    en_count = sum(
        candidate.language == "en"
        for candidate in candidates
    )

    assert az_count == 800
    assert en_count == 200


def test_review_status_is_pending() -> None:
    """New candidates should start in pending review state."""

    config = load_generation_config(
        CONFIG_PATH
    )

    candidates = build_candidates(
        config
    )

    assert all(
        candidate.review_status == "pending"
        for candidate in candidates
    )


def test_invalid_total_quota_raises() -> None:
    """Quota validation should reject an incorrect total."""

    config = load_generation_config(
        CONFIG_PATH
    )

    broken = dict(config)

    broken["total_records"] = 999

    with pytest.raises(ValueError):
        validate_quotas(broken)
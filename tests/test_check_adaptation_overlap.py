from pathlib import Path

import pytest

from src.data.check_adaptation_overlap import (
    build_overlap_report,
    find_exact_overlaps,
    find_high_similarity_overlaps,
    jaccard_similarity,
    load_jsonl,
    normalize_text,
    save_report,
)


def test_normalize_text() -> None:
    text = "Salam, DÜNYA! 123"

    result = normalize_text(
        text
    )

    assert result == "salam dünya 123"


def test_jaccard_similarity_identical() -> None:
    score = jaccard_similarity(
        "bir iki üç",
        "bir iki üç",
    )

    assert score == 1.0


def test_jaccard_similarity_partial() -> None:
    score = jaccard_similarity(
        "bir iki üç",
        "iki üç dörd",
    )

    assert score == 0.5


def test_find_exact_overlaps() -> None:
    adaptation_rows = [
        {
            "item_id": "adapt_00001",
            "instruction": "Azərbaycanın paytaxtı hansıdır?",
        }
    ]

    benchmark_rows = [
        {
            "item_id": "benchmark_00001_az",
            "question": "Azərbaycanın paytaxtı hansıdır?",
        }
    ]

    matches = find_exact_overlaps(
        adaptation_rows=adaptation_rows,
        benchmark_rows=benchmark_rows,
    )

    assert len(matches) == 1

    assert (
        matches[0]["adaptation_item_id"]
        == "adapt_00001"
    )

    assert (
        matches[0]["benchmark_item_id"]
        == "benchmark_00001_az"
    )

    assert (
        matches[0]["match_type"]
        == "exact_normalized"
    )

    assert matches[0]["similarity"] == 1.0


def test_find_exact_overlaps_normalizes_text() -> None:
    adaptation_rows = [
        {
            "item_id": "adapt_00001",
            "instruction": "  SALAM, dünya! ",
        }
    ]

    benchmark_rows = [
        {
            "item_id": "benchmark_00001_az",
            "question": "Salam dünya",
        }
    ]

    matches = find_exact_overlaps(
        adaptation_rows=adaptation_rows,
        benchmark_rows=benchmark_rows,
    )

    assert len(matches) == 1


def test_find_high_similarity_overlaps() -> None:
    adaptation_rows = [
        {
            "item_id": "adapt_00001",
            "instruction": (
                "Aysel kitab mağazasından kitab aldı. "
                "Kitabın qiyməti neçə manat idi?"
            ),
        }
    ]

    benchmark_rows = [
        {
            "item_id": "benchmark_00001_az",
            "question": (
                "Leyla mağazadan kitab aldı. "
                "Kitabın qiyməti neçə manat idi?"
            ),
        }
    ]

    matches = find_high_similarity_overlaps(
        adaptation_rows=adaptation_rows,
        benchmark_rows=benchmark_rows,
        threshold=0.5,
    )

    assert len(matches) == 1

    assert (
        matches[0]["match_type"]
        == "high_token_overlap"
    )

    assert matches[0]["similarity"] >= 0.5


def test_find_high_similarity_skips_exact_match() -> None:
    adaptation_rows = [
        {
            "item_id": "adapt_00001",
            "instruction": "Eyni sual",
        }
    ]

    benchmark_rows = [
        {
            "item_id": "benchmark_00001_az",
            "question": "Eyni sual",
        }
    ]

    matches = find_high_similarity_overlaps(
        adaptation_rows=adaptation_rows,
        benchmark_rows=benchmark_rows,
        threshold=0.0,
    )

    assert matches == []


def test_find_high_similarity_rejects_invalid_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="threshold must be between",
    ):
        find_high_similarity_overlaps(
            adaptation_rows=[],
            benchmark_rows=[],
            threshold=1.5,
        )


def test_build_overlap_report() -> None:
    adaptation_rows = [
        {
            "item_id": "adapt_00001",
            "instruction": "Tam eyni sual",
        },
        {
            "item_id": "adapt_00002",
            "instruction": "Tamamilə başqa mətn",
        },
    ]

    benchmark_rows = [
        {
            "item_id": "benchmark_00001_az",
            "question": "Tam eyni sual",
        }
    ]

    report = build_overlap_report(
        adaptation_rows=adaptation_rows,
        benchmark_rows=benchmark_rows,
        threshold=0.8,
    )

    assert report["adaptation_records"] == 2
    assert report["benchmark_records"] == 1
    assert report["exact_overlap_count"] == 1
    assert report["high_similarity_count"] == 0


def test_load_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data.jsonl"

    path.write_text(
        '{"item_id":"x","instruction":"sual"}\n',
        encoding="utf-8",
    )

    rows = load_jsonl(
        path
    )

    assert len(rows) == 1
    assert rows[0]["item_id"] == "x"


def test_save_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"

    report = {
        "exact_overlap_count": 0,
        "high_similarity_count": 0,
    }

    save_report(
        report=report,
        output_path=path,
    )

    assert path.exists()

    text = path.read_text(
        encoding="utf-8",
    )

    assert '"exact_overlap_count": 0' in text
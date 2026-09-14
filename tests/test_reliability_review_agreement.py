"""Reliability reviewer agreement analizini test eder."""

import csv
from pathlib import Path

import pytest

from src.analysis.reliability_review_agreement import (
    REVIEW_FIELDS,
    analyze_reviews,
    cohen_kappa,
    load_review_csv,
    validate_labels,
    validate_pair_sets,
)


def write_review_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Test reviewer CSV dosyası oluşturur."""

    fieldnames = [
        "pair_id",
        "question_en",
        "question_az",
        *REVIEW_FIELDS,
        "reviewer_notes",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def make_row(
    pair_id: str,
    label: str,
) -> dict[str, str]:
    """Tüm agreement alanları aynı label olan test row'u oluşturur."""

    row = {
        "pair_id": pair_id,
        "question_en": "Question",
        "question_az": "Sual",
        "reviewer_notes": "",
    }

    for field in REVIEW_FIELDS:
        row[field] = label

    return row


def test_cohen_kappa_perfect_agreement() -> None:
    """İki reviewer tamamen aynı dağılıma sahipse kappa 1 olur."""

    labels1 = [
        "yes",
        "yes",
        "no",
        "no",
    ]

    labels2 = [
        "yes",
        "yes",
        "no",
        "no",
    ]

    assert cohen_kappa(
        labels1,
        labels2,
    ) == pytest.approx(1.0)


def test_cohen_kappa_known_disagreement() -> None:
    """Bilinen küçük örnekte Cohen kappa doğru hesaplanır."""

    labels1 = [
        "yes",
        "yes",
        "no",
        "no",
    ]

    labels2 = [
        "yes",
        "no",
        "no",
        "yes",
    ]

    assert cohen_kappa(
        labels1,
        labels2,
    ) == pytest.approx(0.0)


def test_cohen_kappa_empty_returns_none() -> None:
    """Complete pair yoksa kappa hesaplanamaz."""

    assert cohen_kappa(
        [],
        [],
    ) is None


def test_cohen_kappa_constant_labels_returns_none() -> None:
    """Expected agreement 1 olduğunda kappa tanımsızdır."""

    assert cohen_kappa(
        ["yes", "yes"],
        ["yes", "yes"],
    ) is None


def test_load_review_csv_rejects_duplicate_pair_id(
    tmp_path: Path,
) -> None:
    """Duplicate pair_id reddedilir."""

    path = tmp_path / "review.csv"

    rows = [
        make_row(
            "pair_001",
            "yes",
        ),
        make_row(
            "pair_001",
            "no",
        ),
    ]

    write_review_csv(
        path,
        rows,
    )

    with pytest.raises(
        ValueError,
        match="duplicate pair_id",
    ):
        load_review_csv(
            path
        )


def test_validate_labels_rejects_invalid_label(
    tmp_path: Path,
) -> None:
    """yes/no/uncertain dışındaki review label reddedilir."""

    path = tmp_path / "review.csv"

    row = make_row(
        "pair_001",
        "yes",
    )

    row["category_valid"] = "maybe"

    write_review_csv(
        path,
        [row],
    )

    records = load_review_csv(
        path
    )

    with pytest.raises(
        ValueError,
        match="invalid label",
    ):
        validate_labels(
            records,
            path,
        )


def test_validate_pair_sets_rejects_mismatch() -> None:
    """Reviewer pair setleri farklıysa analysis durur."""

    reviewer1 = {
        "pair_001": {},
    }

    reviewer2 = {
        "pair_002": {},
    }

    with pytest.raises(
        ValueError,
        match="identical pair_id sets",
    ):
        validate_pair_sets(
            reviewer1,
            reviewer2,
        )


def test_analyze_reviews_reports_agreement_and_disagreement(
    tmp_path: Path,
) -> None:
    """Agreement summary ve disagreement listesi doğru oluşturulur."""

    reviewer1_path = (
        tmp_path / "reviewer1.csv"
    )

    reviewer2_path = (
        tmp_path / "reviewer2.csv"
    )

    reviewer1_rows = [
        make_row(
            "pair_001",
            "yes",
        ),
        make_row(
            "pair_002",
            "no",
        ),
    ]

    reviewer2_rows = [
        make_row(
            "pair_001",
            "yes",
        ),
        make_row(
            "pair_002",
            "yes",
        ),
    ]

    write_review_csv(
        reviewer1_path,
        reviewer1_rows,
    )

    write_review_csv(
        reviewer2_path,
        reviewer2_rows,
    )

    result = analyze_reviews(
        reviewer1_path=(
            reviewer1_path
        ),
        reviewer2_path=(
            reviewer2_path
        ),
        expected_pairs=2,
    )

    assert result["pairs"] == 2

    assert len(
        result["summaries"]
    ) == len(
        REVIEW_FIELDS
    )

    first_summary = result[
        "summaries"
    ][0]

    assert (
        first_summary[
            "complete_pairs"
        ]
        == 2
    )

    assert (
        first_summary[
            "agreements"
        ]
        == 1
    )

    assert (
        first_summary[
            "disagreements"
        ]
        == 1
    )

    assert (
        first_summary[
            "raw_agreement"
        ]
        == pytest.approx(0.5)
    )

    assert len(
        result["disagreements"]
    ) == len(
        REVIEW_FIELDS
    )


def test_analyze_reviews_handles_missing_labels(
    tmp_path: Path,
) -> None:
    """Eksik reviewer label'ı complete-pair hesabından çıkarılır."""

    reviewer1_path = (
        tmp_path / "reviewer1.csv"
    )

    reviewer2_path = (
        tmp_path / "reviewer2.csv"
    )

    row1 = make_row(
        "pair_001",
        "yes",
    )

    row2 = make_row(
        "pair_001",
        "yes",
    )

    row2["difficulty_valid"] = ""

    write_review_csv(
        reviewer1_path,
        [row1],
    )

    write_review_csv(
        reviewer2_path,
        [row2],
    )

    result = analyze_reviews(
        reviewer1_path=(
            reviewer1_path
        ),
        reviewer2_path=(
            reviewer2_path
        ),
        expected_pairs=1,
    )

    difficulty = next(
        row
        for row in result[
            "summaries"
        ]
        if row["field"]
        == "difficulty_valid"
    )

    assert (
        difficulty[
            "complete_pairs"
        ]
        == 0
    )

    assert (
        difficulty[
            "missing_pairs"
        ]
        == 1
    )

    assert (
        difficulty[
            "raw_agreement"
        ]
        is None
    )

    assert (
        difficulty[
            "cohen_kappa"
        ]
        is None
    )

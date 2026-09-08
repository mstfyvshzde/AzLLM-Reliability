from pathlib import Path

import pytest

from src.data.review_adaptation_data import (
    build_review_summary,
    get_approved_records,
    load_records,
    save_records,
    set_all_review_status,
    set_review_status,
    validate_review_status,
)


def make_record(
    item_id: str,
    status: str = "pending",
) -> dict:
    return {
        "item_id": item_id,
        "language": "az",
        "category": "reasoning",
        "instruction": "Sadə test sualı",
        "response": "Bəli.",
        "source": "synthetic",
        "metadata": {
            "generation_status": "generated",
            "review_status": status,
            "subcategory": "logical_reasoning",
        },
    }


def test_validate_review_status_passes() -> None:
    validate_review_status(
        "pending"
    )
    validate_review_status(
        "approved"
    )
    validate_review_status(
        "rejected"
    )


def test_validate_review_status_rejects_invalid() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid review status",
    ):
        validate_review_status(
            "unknown"
        )


def test_set_review_status_updates_record() -> None:
    records = [
        make_record(
            "adapt_00001"
        ),
        make_record(
            "adapt_00002"
        ),
    ]

    updated = set_review_status(
        records=records,
        item_id="adapt_00002",
        status="approved",
    )

    assert (
        updated[0]["metadata"]["review_status"]
        == "pending"
    )

    assert (
        updated[1]["metadata"]["review_status"]
        == "approved"
    )


def test_set_review_status_preserves_original_records() -> None:
    records = [
        make_record(
            "adapt_00001"
        )
    ]

    updated = set_review_status(
        records=records,
        item_id="adapt_00001",
        status="approved",
    )

    assert (
        records[0]["metadata"]["review_status"]
        == "pending"
    )

    assert (
        updated[0]["metadata"]["review_status"]
        == "approved"
    )


def test_set_review_status_rejects_unknown_item() -> None:
    records = [
        make_record(
            "adapt_00001"
        )
    ]

    with pytest.raises(
        ValueError,
        match="Unknown item_id",
    ):
        set_review_status(
            records=records,
            item_id="adapt_99999",
            status="approved",
        )


def test_set_all_review_status() -> None:
    records = [
        make_record(
            "adapt_00001"
        ),
        make_record(
            "adapt_00002"
        ),
    ]

    updated = set_all_review_status(
        records=records,
        status="approved",
    )

    assert all(
        record["metadata"]["review_status"]
        == "approved"
        for record in updated
    )


def test_get_approved_records() -> None:
    records = [
        make_record(
            "adapt_00001",
            status="approved",
        ),
        make_record(
            "adapt_00002",
            status="rejected",
        ),
        make_record(
            "adapt_00003",
            status="pending",
        ),
    ]

    approved = get_approved_records(
        records
    )

    assert len(approved) == 1
    assert approved[0]["item_id"] == "adapt_00001"


def test_build_review_summary() -> None:
    records = [
        make_record(
            "adapt_00001",
            status="approved",
        ),
        make_record(
            "adapt_00002",
            status="approved",
        ),
        make_record(
            "adapt_00003",
            status="rejected",
        ),
        make_record(
            "adapt_00004",
            status="pending",
        ),
    ]

    summary = build_review_summary(
        records
    )

    assert summary == {
        "total": 4,
        "approved": 2,
        "rejected": 1,
        "pending": 1,
    }


def test_build_review_summary_rejects_invalid_status() -> None:
    records = [
        make_record(
            "adapt_00001"
        )
    ]

    records[0]["metadata"][
        "review_status"
    ] = "invalid"

    with pytest.raises(
        ValueError,
        match="Invalid or missing review_status",
    ):
        build_review_summary(
            records
        )


def test_load_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"

    path.write_text(
        (
            '{"item_id":"adapt_00001",'
            '"metadata":{"review_status":"pending"}}\n'
        ),
        encoding="utf-8",
    )

    records = load_records(
        path
    )

    assert len(records) == 1
    assert (
        records[0]["item_id"]
        == "adapt_00001"
    )


def test_save_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "output.jsonl"

    records = [
        make_record(
            "adapt_00001"
        )
    ]

    save_records(
        records=records,
        output_path=path,
    )

    assert path.exists()

    loaded = load_records(
        path
    )

    assert len(loaded) == 1

    assert (
        loaded[0]["item_id"]
        == "adapt_00001"
    )


def test_save_records_rejects_existing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "output.jsonl"

    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="Output already exists",
    ):
        save_records(
            records=[
                make_record(
                    "adapt_00001"
                )
            ],
            output_path=path,
        )
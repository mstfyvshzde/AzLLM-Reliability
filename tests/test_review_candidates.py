"""Candidate review durumlarını ve approved pair filtrelemesini test eder."""

from src.data.benchmark_record import BenchmarkRecord
from src.data.review_candidates import (
    get_approved_records,
    set_pair_review_status
)


def make_record(
    item_id: str,
    pair_id: str,
    language: str,
    status: str = "pending"
) -> BenchmarkRecord:
    """Review testleri için örnek benchmark kaydı oluşturur."""
    return BenchmarkRecord(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task="reasoning",
        question="Example question",
        reference_answer="Example answer",
        metadata={"review_status": status}
    )


def test_set_pair_review_status() -> None:
    """Bir pair içindeki tüm kayıtların review durumunun değiştiğini test eder."""
    records = [
        make_record("reasoning_0001_en", "reasoning_0001", "en"),
        make_record("reasoning_0001_az", "reasoning_0001", "az")
    ]

    updated_records = set_pair_review_status(
        records,
        pair_id="reasoning_0001",
        status="approved"
    )

    for record in updated_records:
        assert record.metadata["review_status"] == "approved"


def test_unknown_pair_is_rejected() -> None:
    """Bulunmayan pair_id değerinin reddedildiğini test eder."""
    records = [
        make_record("reasoning_0001_en", "reasoning_0001", "en"),
        make_record("reasoning_0001_az", "reasoning_0001", "az")
    ]

    try:
        set_pair_review_status(
            records,
            pair_id="reasoning_9999",
            status="approved",
        )
    except ValueError as error:
        assert "Pair not found" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")


def test_invalid_review_status_is_rejected() -> None:
    """Geçersiz review status değerinin reddedildiğini test eder."""
    records = [
        make_record("reasoning_0001_en", "reasoning_0001", "en"),
        make_record("reasoning_0001_az", "reasoning_0001", "az")
    ]

    try:
        set_pair_review_status(
            records,
            pair_id="reasoning_0001",
            status="invalid"
        )
    except ValueError as error:
        assert "Invalid review status" in str(error)
    else:
        raise AssertionError("Expected ValueError was not raised.")


def test_get_approved_records() -> None:
    """Yalnızca tamamen approved olan pair kayıtlarının döndürüldüğünü test eder."""
    records = [
        make_record(
            "reasoning_0001_en",
            "reasoning_0001",
            "en",
            status="approved"
        ),
        make_record(
            "reasoning_0001_az",
            "reasoning_0001",
            "az",
            status="approved"
        ),
        make_record(
            "reasoning_0002_en",
            "reasoning_0002",
            "en",
            status="pending"
        ),
        make_record(
            "reasoning_0002_az",
            "reasoning_0002",
            "az",
            status="pending"
        )
    ]

    approved_records = get_approved_records(records)

    assert len(approved_records) == 2

    for record in approved_records:
        assert record.pair_id == "reasoning_0001"
        assert record.metadata["review_status"] == "approved"


def test_partially_approved_pair_is_not_returned() -> None:
    """Tek dili approved olan eksik review pair'inin final listeye girmediğini test eder."""
    records = [
        make_record(
            "reasoning_0001_en",
            "reasoning_0001",
            "en",
            status="approved"
        ),
        make_record(
            "reasoning_0001_az",
            "reasoning_0001",
            "az",
            status="pending"
        )
    ]

    approved_records = get_approved_records(records)

    assert approved_records == []
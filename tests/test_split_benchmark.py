"""Pair-aware train-dev-test benchmark split işlemlerini test eder."""

from src.data.benchmark_record import BenchmarkRecord
from src.data.split_benchmark import (
    calculate_split_sizes,
    split_records_by_pair,
    validate_split_integrity,
    validate_split_ratios,
)


def make_pair(
    pair_index: int,
) -> list[BenchmarkRecord]:
    """Split testleri için örnek EN-AZ benchmark pair'i oluşturur."""

    pair_id = f"reasoning_{pair_index:04d}"

    return [
        BenchmarkRecord(
            item_id=f"{pair_id}_en",
            pair_id=pair_id,
            language="en",
            task="reasoning",
            question=f"English question {pair_index}",
            reference_answer="Answer",
            metadata={
                "category": "arithmetic_reasoning",
                "difficulty": "easy",
                "review_status": "approved",
            },
        ),
        BenchmarkRecord(
            item_id=f"{pair_id}_az",
            pair_id=pair_id,
            language="az",
            task="reasoning",
            question=f"Azerbaijani question {pair_index}",
            reference_answer="Answer",
            metadata={
                "category": "arithmetic_reasoning",
                "difficulty": "easy",
                "review_status": "approved",
            },
        ),
    ]


def make_records(
    pair_count: int,
) -> list[BenchmarkRecord]:
    """Belirtilen sayıda EN-AZ benchmark pair'i oluşturur."""

    records: list[BenchmarkRecord] = []

    for pair_index in range(1, pair_count + 1):
        records.extend(
            make_pair(pair_index)
        )

    return records


def test_valid_split_ratios() -> None:
    """Toplamı 1.0 olan geçerli split oranlarının kabul edildiğini test eder."""

    validate_split_ratios(
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
    )


def test_invalid_split_ratio_sum_is_rejected() -> None:
    """Toplamı 1.0 olmayan split oranlarının reddedildiğini test eder."""

    try:
        validate_split_ratios(
            train_ratio=0.70,
            dev_ratio=0.20,
            test_ratio=0.20,
        )
    except ValueError as error:
        assert "must sum to 1.0" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError was not raised."
        )


def test_out_of_range_split_ratio_is_rejected() -> None:
    """0-1 aralığı dışında kalan split oranının reddedildiğini test eder."""

    try:
        validate_split_ratios(
            train_ratio=1.10,
            dev_ratio=0.00,
            test_ratio=-0.10,
        )
    except ValueError as error:
        assert "ratio must be between 0 and 1" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError was not raised."
        )


def test_calculate_split_sizes() -> None:
    """Pair sayısından beklenen train-dev-test boyutlarının üretildiğini test eder."""

    train_size, dev_size, test_size = calculate_split_sizes(
        pair_count=20,
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
    )

    assert train_size == 14
    assert dev_size == 3
    assert test_size == 3


def test_split_records_by_pair() -> None:
    """Benchmark pair'lerinin train-dev-test split'lerine ayrıldığını test eder."""

    records = make_records(
        pair_count=20
    )

    splits = split_records_by_pair(
        records=records,
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
        seed=17,
    )

    assert len(splits["train"]) == 28
    assert len(splits["dev"]) == 6
    assert len(splits["test"]) == 6


def test_pair_records_remain_in_same_split() -> None:
    """Aynı pair içindeki EN ve AZ kayıtlarının aynı split'te kaldığını test eder."""

    records = make_records(
        pair_count=20
    )

    splits = split_records_by_pair(
        records=records,
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
        seed=17,
    )

    pair_locations: dict[str, str] = {}

    for split_name, split_records in splits.items():
        for record in split_records:
            previous_location = pair_locations.get(
                record.pair_id
            )

            if previous_location is not None:
                assert previous_location == split_name

            pair_locations[record.pair_id] = split_name


def test_split_is_reproducible() -> None:
    """Aynı seed ile aynı pair split'lerinin üretildiğini test eder."""

    records = make_records(
        pair_count=20
    )

    first_splits = split_records_by_pair(
        records=records,
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
        seed=17,
    )

    second_splits = split_records_by_pair(
        records=records,
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
        seed=17,
    )

    for split_name in (
        "train",
        "dev",
        "test",
    ):
        first_ids = [
            record.item_id
            for record in first_splits[split_name]
        ]

        second_ids = [
            record.item_id
            for record in second_splits[split_name]
        ]

        assert first_ids == second_ids


def test_different_seed_can_change_split_assignment() -> None:
    """Farklı seed değerlerinin pair sıralamasını değiştirebildiğini test eder."""

    records = make_records(
        pair_count=20
    )

    first_splits = split_records_by_pair(
        records=records,
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
        seed=17,
    )

    second_splits = split_records_by_pair(
        records=records,
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
        seed=99,
    )

    first_train_pairs = {
        record.pair_id
        for record in first_splits["train"]
    }

    second_train_pairs = {
        record.pair_id
        for record in second_splits["train"]
    }

    assert first_train_pairs != second_train_pairs


def test_split_integrity() -> None:
    """Geçerli split yapısının integrity kontrolünden geçtiğini test eder."""

    records = make_records(
        pair_count=20
    )

    splits = split_records_by_pair(
        records=records,
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
        seed=17,
    )

    validate_split_integrity(
        records,
        splits,
    )


def test_split_integrity_rejects_pair_leakage() -> None:
    """Aynı pair farklı split'lere düşerse integrity kontrolünün reddettiğini test eder."""

    records = make_pair(
        pair_index=1
    )

    splits = {
        "train": [
            records[0],
        ],
        "dev": [
            records[1],
        ],
        "test": [],
    }

    try:
        validate_split_integrity(
            records,
            splits,
        )
    except ValueError as error:
        assert "appears in both" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError was not raised."
        )


def test_split_integrity_rejects_missing_record() -> None:
    """Split sırasında bir record kaybolursa integrity kontrolünün reddettiğini test eder."""

    records = make_pair(
        pair_index=1
    )

    splits = {
        "train": [
            records[0],
        ],
        "dev": [],
        "test": [],
    }

    try:
        validate_split_integrity(
            records,
            splits,
        )
    except ValueError as error:
        assert "do not match original benchmark records" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError was not raised."
        )
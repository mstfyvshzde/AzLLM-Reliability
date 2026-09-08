from src.data.adaptation_record import AdaptationRecord


def test_adaptation_record_to_dict() -> None:
    record = AdaptationRecord(
        item_id="adapt_0001",
        language="az",
        category="instruction_following",
        instruction="Yalnız nəticəni qaytarın.",
        response="42",
        source="synthetic",
        metadata={
            "review_status": "pending",
        },
    )

    result = record.to_dict()

    assert result["item_id"] == "adapt_0001"
    assert result["language"] == "az"
    assert result["category"] == "instruction_following"
    assert result["instruction"] == "Yalnız nəticəni qaytarın."
    assert result["response"] == "42"
    assert result["source"] == "synthetic"
    assert result["metadata"]["review_status"] == "pending"

"""Response-status reliability sınıflandırmasını test eder."""

import pytest

from src.evaluation.run_inference import PredictionRecord
from src.reliability.response_status import (
    ABSTAINED_STATUS,
    ANSWERED_STATUS,
    EMPTY_STATUS,
    ResponseStatusResult,
    classify_response_status,
    contains_abstention_signal,
    create_response_status_result,
    evaluate_response_statuses,
    is_empty_response,
    normalize_response_text,
    summarize_response_statuses,
)


def make_prediction_record(
    prediction: str,
    item_id: str = "reasoning_0001_en",
    pair_id: str = "reasoning_0001",
    language: str = "en",
    task: str = "reasoning",
) -> PredictionRecord:
    """Response-status testleri için örnek PredictionRecord oluşturur."""

    return PredictionRecord(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        question="What is the answer?",
        reference_answer="4",
        prediction=prediction,
        metadata={
            "category": "arithmetic_reasoning",
            "difficulty": "easy",
            "review_status": "approved",
        },
    )


def test_normalize_response_text() -> None:
    """Metnin lowercase ve whitespace açısından normalize edildiğini test eder."""

    result = normalize_response_text(
        "  I   DO NOT\nKNOW  "
    )

    assert result == "i do not know"


def test_normalize_response_text_rejects_non_string() -> None:
    """String olmayan response değerinin reddedildiğini test eder."""

    with pytest.raises(
        TypeError,
        match="Response text must be a string",
    ):
        normalize_response_text(
            42  # type: ignore[arg-type]
        )


def test_is_empty_response_for_empty_string() -> None:
    """Boş string response'un empty olarak algılandığını test eder."""

    assert is_empty_response(
        ""
    )


def test_is_empty_response_for_whitespace_only() -> None:
    """Yalnızca whitespace içeren response'un empty kabul edildiğini test eder."""

    assert is_empty_response(
        "   \n\t  "
    )


def test_is_empty_response_for_non_empty_text() -> None:
    """Normal text response'un empty olmadığını test eder."""

    assert not is_empty_response(
        "4"
    )


def test_contains_english_abstention_signal() -> None:
    """İngilizce açık abstention ifadesinin yakalandığını test eder."""

    assert contains_abstention_signal(
        "I don't know the answer."
    )


def test_contains_english_insufficient_information_signal() -> None:
    """İngilizce insufficient-information ifadesinin yakalandığını test eder."""

    assert contains_abstention_signal(
        "There is not enough information to determine the answer."
    )


def test_contains_azerbaijani_abstention_signal() -> None:
    """Azerbaycanca açık abstention ifadesinin yakalandığını test eder."""

    assert contains_abstention_signal(
        "Bu suala cavab verə bilmirəm."
    )


def test_contains_azerbaijani_insufficient_information_signal() -> None:
    """Azerbaycanca yetersiz bilgi ifadesinin yakalandığını test eder."""

    assert contains_abstention_signal(
        "Kifayət qədər məlumat yoxdur."
    )


def test_normal_answer_does_not_trigger_abstention() -> None:
    """Normal cevabın yanlışlıkla abstention olarak sınıflandırılmadığını test eder."""

    assert not contains_abstention_signal(
        "The answer is 4."
    )


def test_classify_empty_response() -> None:
    """Boş prediction için empty status döndüğünü test eder."""

    assert classify_response_status(
        "   "
    ) == EMPTY_STATUS


def test_classify_abstained_response() -> None:
    """Abstention sinyali taşıyan prediction için abstained döndüğünü test eder."""

    assert classify_response_status(
        "I am not sure."
    ) == ABSTAINED_STATUS


def test_classify_answered_response() -> None:
    """Normal prediction için answered status döndüğünü test eder."""

    assert classify_response_status(
        "The answer is 4."
    ) == ANSWERED_STATUS


def test_create_response_status_result() -> None:
    """PredictionRecord üzerinden item-level response-status sonucu üretildiğini test eder."""

    record = make_prediction_record(
        prediction="I don't know."
    )

    result = create_response_status_result(
        record
    )

    assert isinstance(
        result,
        ResponseStatusResult,
    )

    assert result.item_id == record.item_id
    assert result.pair_id == record.pair_id
    assert result.language == record.language
    assert result.task == record.task
    assert result.prediction == record.prediction
    assert result.status == ABSTAINED_STATUS
    assert result.metadata == record.metadata


def test_response_status_result_to_dict() -> None:
    """ResponseStatusResult nesnesinin sözlüğe doğru dönüştürüldüğünü test eder."""

    result = ResponseStatusResult(
        item_id="item_1",
        pair_id="pair_1",
        language="en",
        task="reasoning",
        prediction="4",
        status=ANSWERED_STATUS,
        metadata={
            "difficulty": "easy",
        },
    )

    assert result.to_dict() == {
        "item_id": "item_1",
        "pair_id": "pair_1",
        "language": "en",
        "task": "reasoning",
        "prediction": "4",
        "status": "answered",
        "metadata": {
            "difficulty": "easy",
        },
    }


def test_evaluate_response_statuses() -> None:
    """Birden fazla prediction kaydının topluca sınıflandırıldığını test eder."""

    records = [
        make_prediction_record(
            prediction="4",
            item_id="item_1",
        ),
        make_prediction_record(
            prediction="I don't know.",
            item_id="item_2",
        ),
        make_prediction_record(
            prediction="   ",
            item_id="item_3",
        ),
    ]

    results = evaluate_response_statuses(
        records
    )

    assert [
        result.status
        for result in results
    ] == [
        ANSWERED_STATUS,
        ABSTAINED_STATUS,
        EMPTY_STATUS,
    ]


def test_evaluate_response_statuses_preserves_input_order() -> None:
    """Batch evaluation sırasında input sırasının korunduğunu test eder."""

    records = [
        make_prediction_record(
            prediction="First answer",
            item_id="item_2",
        ),
        make_prediction_record(
            prediction="Second answer",
            item_id="item_1",
        ),
    ]

    results = evaluate_response_statuses(
        records
    )

    assert [
        result.item_id
        for result in results
    ] == [
        "item_2",
        "item_1",
    ]


def test_evaluate_response_statuses_empty_input() -> None:
    """Boş prediction listesinin boş result listesi döndürdüğünü test eder."""

    assert evaluate_response_statuses(
        []
    ) == []


def test_summarize_response_statuses() -> None:
    """Aggregate response-status sayı ve oranlarının doğru hesaplandığını test eder."""

    results = [
        ResponseStatusResult(
            item_id="item_1",
            pair_id="pair_1",
            language="en",
            task="reasoning",
            prediction="4",
            status=ANSWERED_STATUS,
            metadata={},
        ),
        ResponseStatusResult(
            item_id="item_2",
            pair_id="pair_2",
            language="en",
            task="reasoning",
            prediction="I don't know.",
            status=ABSTAINED_STATUS,
            metadata={},
        ),
        ResponseStatusResult(
            item_id="item_3",
            pair_id="pair_3",
            language="az",
            task="reasoning",
            prediction="",
            status=EMPTY_STATUS,
            metadata={},
        ),
        ResponseStatusResult(
            item_id="item_4",
            pair_id="pair_4",
            language="az",
            task="reasoning",
            prediction="5",
            status=ANSWERED_STATUS,
            metadata={},
        ),
    ]

    summary = summarize_response_statuses(
        results
    )

    assert summary["total"] == 4
    assert summary["answered"] == 2
    assert summary["abstained"] == 1
    assert summary["empty"] == 1

    assert summary["answered_rate"] == pytest.approx(
        0.5
    )

    assert summary["abstention_rate"] == pytest.approx(
        0.25
    )

    assert summary["empty_rate"] == pytest.approx(
        0.25
    )


def test_summarize_response_statuses_rejects_empty_input() -> None:
    """Boş result listesinin summarize edilmesinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Cannot summarize empty response-status results",
    ):
        summarize_response_statuses(
            []
        )


def test_summarize_response_statuses_rejects_unknown_status() -> None:
    """Tanımsız response status değerinin aggregate analizde reddedildiğini test eder."""

    result = ResponseStatusResult(
        item_id="item_1",
        pair_id="pair_1",
        language="en",
        task="reasoning",
        prediction="4",
        status="unknown",
        metadata={},
    )

    with pytest.raises(
        ValueError,
        match="Unknown response status",
    ):
        summarize_response_statuses(
            [result]
        )
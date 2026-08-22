"""Abstention reliability metric yardımcılarını test eder."""

import pytest

from src.evaluation.run_inference import PredictionRecord
from src.reliability.abstention_metrics import (
    CORRECT_ABSTENTION,
    CORRECT_ANSWER,
    EMPTY_RESPONSE,
    OVER_ANSWERING,
    UNDER_ANSWERING,
    AbstentionResult,
    classify_abstention_outcome,
    create_abstention_result,
    evaluate_abstention,
    get_answerability,
    summarize_abstention_results,
)
from src.reliability.response_status import (
    ABSTAINED_STATUS,
    ANSWERED_STATUS,
    EMPTY_STATUS,
)


def make_prediction_record(
    prediction: str,
    is_answerable: bool,
    item_id: str = "item_1",
    pair_id: str = "pair_1",
    language: str = "en",
    task: str = "reasoning",
) -> PredictionRecord:
    """Abstention metric testleri için örnek PredictionRecord oluşturur."""

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
            "is_answerable": is_answerable,
        },
    )


def test_classify_correct_answer() -> None:
    """Answerable item üzerinde cevap verilmesini correct_answer olarak sınıflandırır."""

    assert classify_abstention_outcome(
        is_answerable=True,
        response_status=ANSWERED_STATUS,
    ) == CORRECT_ANSWER


def test_classify_under_answering() -> None:
    """Answerable item üzerinde abstain edilmesini under_answering olarak sınıflandırır."""

    assert classify_abstention_outcome(
        is_answerable=True,
        response_status=ABSTAINED_STATUS,
    ) == UNDER_ANSWERING


def test_classify_correct_abstention() -> None:
    """Unanswerable item üzerinde abstain edilmesini correct_abstention olarak sınıflandırır."""

    assert classify_abstention_outcome(
        is_answerable=False,
        response_status=ABSTAINED_STATUS,
    ) == CORRECT_ABSTENTION


def test_classify_over_answering() -> None:
    """Unanswerable item üzerinde cevap verilmesini over_answering olarak sınıflandırır."""

    assert classify_abstention_outcome(
        is_answerable=False,
        response_status=ANSWERED_STATUS,
    ) == OVER_ANSWERING


def test_classify_empty_response_for_answerable_item() -> None:
    """Answerable item üzerindeki empty response'un ayrı tutulduğunu test eder."""

    assert classify_abstention_outcome(
        is_answerable=True,
        response_status=EMPTY_STATUS,
    ) == EMPTY_RESPONSE


def test_classify_empty_response_for_unanswerable_item() -> None:
    """Unanswerable item üzerindeki empty response'un ayrı tutulduğunu test eder."""

    assert classify_abstention_outcome(
        is_answerable=False,
        response_status=EMPTY_STATUS,
    ) == EMPTY_RESPONSE


def test_classify_rejects_unknown_response_status() -> None:
    """Tanımsız response-status değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Unknown response status",
    ):
        classify_abstention_outcome(
            is_answerable=True,
            response_status="unknown",
        )


def test_get_answerability_true() -> None:
    """Metadata içindeki True answerability değerinin okunduğunu test eder."""

    record = make_prediction_record(
        prediction="4",
        is_answerable=True,
    )

    assert get_answerability(
        record
    ) is True


def test_get_answerability_false() -> None:
    """Metadata içindeki False answerability değerinin okunduğunu test eder."""

    record = make_prediction_record(
        prediction="I don't know.",
        is_answerable=False,
    )

    assert get_answerability(
        record
    ) is False


def test_get_answerability_rejects_missing_metadata() -> None:
    """is_answerable metadata alanı eksik olduğunda hata oluşturulduğunu test eder."""

    record = make_prediction_record(
        prediction="4",
        is_answerable=True,
    )

    record.metadata.pop(
        "is_answerable"
    )

    with pytest.raises(
        KeyError,
        match="Missing 'is_answerable' metadata",
    ):
        get_answerability(
            record
        )


def test_get_answerability_rejects_non_boolean_value() -> None:
    """is_answerable bool değilse reddedildiğini test eder."""

    record = make_prediction_record(
        prediction="4",
        is_answerable=True,
    )

    record.metadata[
        "is_answerable"
    ] = "yes"

    with pytest.raises(
        TypeError,
        match="'is_answerable' must be bool",
    ):
        get_answerability(
            record
        )


def test_create_correct_answer_result() -> None:
    """Answerable item ve normal cevap için correct_answer sonucu üretildiğini test eder."""

    record = make_prediction_record(
        prediction="4",
        is_answerable=True,
    )

    result = create_abstention_result(
        record
    )

    assert isinstance(
        result,
        AbstentionResult,
    )

    assert result.item_id == record.item_id
    assert result.pair_id == record.pair_id
    assert result.language == record.language
    assert result.task == record.task
    assert result.prediction == record.prediction

    assert result.is_answerable is True
    assert result.response_status == ANSWERED_STATUS
    assert result.outcome == CORRECT_ANSWER
    assert result.metadata == record.metadata


def test_create_correct_abstention_result() -> None:
    """Unanswerable item üzerinde abstention sonucunun doğru üretildiğini test eder."""

    record = make_prediction_record(
        prediction="I don't know.",
        is_answerable=False,
    )

    result = create_abstention_result(
        record
    )

    assert result.response_status == ABSTAINED_STATUS
    assert result.outcome == CORRECT_ABSTENTION


def test_create_over_answering_result() -> None:
    """Unanswerable item üzerinde cevap verilmesini over_answering olarak işaretler."""

    record = make_prediction_record(
        prediction="The answer is 4.",
        is_answerable=False,
    )

    result = create_abstention_result(
        record
    )

    assert result.outcome == OVER_ANSWERING


def test_create_under_answering_result() -> None:
    """Answerable item üzerinde abstention yapılmasını under_answering olarak işaretler."""

    record = make_prediction_record(
        prediction="I am not sure.",
        is_answerable=True,
    )

    result = create_abstention_result(
        record
    )

    assert result.outcome == UNDER_ANSWERING


def test_create_empty_response_result() -> None:
    """Boş prediction için empty_response outcome üretildiğini test eder."""

    record = make_prediction_record(
        prediction="   ",
        is_answerable=True,
    )

    result = create_abstention_result(
        record
    )

    assert result.response_status == EMPTY_STATUS
    assert result.outcome == EMPTY_RESPONSE


def test_abstention_result_to_dict() -> None:
    """AbstentionResult nesnesinin sözlüğe doğru dönüştürüldüğünü test eder."""

    result = AbstentionResult(
        item_id="item_1",
        pair_id="pair_1",
        language="en",
        task="reasoning",
        prediction="4",
        is_answerable=True,
        response_status=ANSWERED_STATUS,
        outcome=CORRECT_ANSWER,
        metadata={
            "is_answerable": True,
        },
    )

    assert result.to_dict() == {
        "item_id": "item_1",
        "pair_id": "pair_1",
        "language": "en",
        "task": "reasoning",
        "prediction": "4",
        "is_answerable": True,
        "response_status": "answered",
        "outcome": "correct_answer",
        "metadata": {
            "is_answerable": True,
        },
    }


def test_evaluate_abstention() -> None:
    """Prediction kayıtlarının topluca abstention reliability değerlendirmesini test eder."""

    records = [
        make_prediction_record(
            prediction="4",
            is_answerable=True,
            item_id="item_1",
        ),
        make_prediction_record(
            prediction="I don't know.",
            is_answerable=False,
            item_id="item_2",
        ),
        make_prediction_record(
            prediction="Paris.",
            is_answerable=False,
            item_id="item_3",
        ),
        make_prediction_record(
            prediction="I cannot answer.",
            is_answerable=True,
            item_id="item_4",
        ),
    ]

    results = evaluate_abstention(
        records
    )

    assert [
        result.outcome
        for result in results
    ] == [
        CORRECT_ANSWER,
        CORRECT_ABSTENTION,
        OVER_ANSWERING,
        UNDER_ANSWERING,
    ]


def test_evaluate_abstention_preserves_input_order() -> None:
    """Batch abstention evaluation sırasında input sırasının korunduğunu test eder."""

    records = [
        make_prediction_record(
            prediction="4",
            is_answerable=True,
            item_id="item_2",
        ),
        make_prediction_record(
            prediction="5",
            is_answerable=True,
            item_id="item_1",
        ),
    ]

    results = evaluate_abstention(
        records
    )

    assert [
        result.item_id
        for result in results
    ] == [
        "item_2",
        "item_1",
    ]


def test_evaluate_abstention_empty_input() -> None:
    """Boş prediction listesinin boş abstention result listesi döndürdüğünü test eder."""

    assert evaluate_abstention(
        []
    ) == []


def test_summarize_abstention_results() -> None:
    """Abstention outcome sayı ve oranlarının doğru hesaplandığını test eder."""

    results = [
        AbstentionResult(
            item_id="item_1",
            pair_id="pair_1",
            language="en",
            task="reasoning",
            prediction="4",
            is_answerable=True,
            response_status=ANSWERED_STATUS,
            outcome=CORRECT_ANSWER,
            metadata={},
        ),
        AbstentionResult(
            item_id="item_2",
            pair_id="pair_2",
            language="en",
            task="reasoning",
            prediction="I don't know.",
            is_answerable=True,
            response_status=ABSTAINED_STATUS,
            outcome=UNDER_ANSWERING,
            metadata={},
        ),
        AbstentionResult(
            item_id="item_3",
            pair_id="pair_3",
            language="az",
            task="reasoning",
            prediction="Bilmirəm.",
            is_answerable=False,
            response_status=ABSTAINED_STATUS,
            outcome=CORRECT_ABSTENTION,
            metadata={},
        ),
        AbstentionResult(
            item_id="item_4",
            pair_id="pair_4",
            language="az",
            task="reasoning",
            prediction="42",
            is_answerable=False,
            response_status=ANSWERED_STATUS,
            outcome=OVER_ANSWERING,
            metadata={},
        ),
    ]

    summary = summarize_abstention_results(
        results
    )

    assert summary["total"] == 4
    assert summary["correct_answer"] == 1
    assert summary["under_answering"] == 1
    assert summary["correct_abstention"] == 1
    assert summary["over_answering"] == 1
    assert summary["empty_response"] == 0

    assert summary["abstention_accuracy"] == pytest.approx(
        0.5
    )

    assert summary["over_answering_rate"] == pytest.approx(
        0.5
    )

    assert summary["under_answering_rate"] == pytest.approx(
        0.5
    )


def test_summarize_with_no_unanswerable_items() -> None:
    """Unanswerable item yokken over-answering rate'in 0.0 olduğunu test eder."""

    results = [
        AbstentionResult(
            item_id="item_1",
            pair_id="pair_1",
            language="en",
            task="reasoning",
            prediction="4",
            is_answerable=True,
            response_status=ANSWERED_STATUS,
            outcome=CORRECT_ANSWER,
            metadata={},
        )
    ]

    summary = summarize_abstention_results(
        results
    )

    assert summary["over_answering_rate"] == 0.0


def test_summarize_with_no_answerable_items() -> None:
    """Answerable item yokken under-answering rate'in 0.0 olduğunu test eder."""

    results = [
        AbstentionResult(
            item_id="item_1",
            pair_id="pair_1",
            language="az",
            task="reasoning",
            prediction="Bilmirəm.",
            is_answerable=False,
            response_status=ABSTAINED_STATUS,
            outcome=CORRECT_ABSTENTION,
            metadata={},
        )
    ]

    summary = summarize_abstention_results(
        results
    )

    assert summary["under_answering_rate"] == 0.0


def test_summarize_abstention_results_rejects_empty_input() -> None:
    """Boş abstention result listesinin summarize edilmesinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Cannot summarize empty abstention results",
    ):
        summarize_abstention_results(
            []
        )


def test_summarize_abstention_results_rejects_unknown_outcome() -> None:
    """Tanımsız abstention outcome değerinin reddedildiğini test eder."""

    result = AbstentionResult(
        item_id="item_1",
        pair_id="pair_1",
        language="en",
        task="reasoning",
        prediction="4",
        is_answerable=True,
        response_status=ANSWERED_STATUS,
        outcome="unknown",
        metadata={},
    )

    with pytest.raises(
        ValueError,
        match="Unknown abstention outcome",
    ):
        summarize_abstention_results(
            [result]
        )
"""Paired EN-AZ reliability metric yardımcılarını test eder."""

import pytest

from src.reliability.abstention_metrics import (
    CORRECT_ABSTENTION,
    CORRECT_ANSWER,
    EMPTY_RESPONSE,
    OVER_ANSWERING,
    UNDER_ANSWERING,
    AbstentionResult,
)
from src.reliability.paired_metrics import (
    PairedReliabilityResult,
    classify_reliability_transition,
    create_paired_reliability_result,
    evaluate_paired_reliability,
    group_reliability_by_pair,
    is_reliable_outcome,
    summarize_paired_reliability,
)


def make_result(
    item_id: str,
    pair_id: str,
    language: str,
    outcome: str,
    is_answerable: bool,
    task: str = "reasoning",
) -> AbstentionResult:
    """Paired reliability testleri için örnek AbstentionResult oluşturur."""

    if outcome in {
        CORRECT_ANSWER,
        OVER_ANSWERING,
    }:
        response_status = "answered"
        prediction = "4"

    elif outcome in {
        CORRECT_ABSTENTION,
        UNDER_ANSWERING,
    }:
        response_status = "abstained"
        prediction = "I don't know."

    else:
        response_status = "empty"
        prediction = ""

    return AbstentionResult(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        prediction=prediction,
        is_answerable=is_answerable,
        response_status=response_status,
        outcome=outcome,
        metadata={
            "category": "arithmetic_reasoning",
            "difficulty": "easy",
            "review_status": "approved",
            "is_answerable": is_answerable,
        },
    )


def test_is_reliable_outcome_for_correct_answer() -> None:
    """correct_answer outcome'un reliable olduğunu test eder."""

    assert is_reliable_outcome(
        CORRECT_ANSWER
    ) is True


def test_is_reliable_outcome_for_correct_abstention() -> None:
    """correct_abstention outcome'un reliable olduğunu test eder."""

    assert is_reliable_outcome(
        CORRECT_ABSTENTION
    ) is True


def test_is_reliable_outcome_for_under_answering() -> None:
    """under_answering outcome'un unreliable olduğunu test eder."""

    assert is_reliable_outcome(
        UNDER_ANSWERING
    ) is False


def test_is_reliable_outcome_for_over_answering() -> None:
    """over_answering outcome'un unreliable olduğunu test eder."""

    assert is_reliable_outcome(
        OVER_ANSWERING
    ) is False


def test_is_reliable_outcome_for_empty_response() -> None:
    """empty_response outcome'un unreliable olduğunu test eder."""

    assert is_reliable_outcome(
        EMPTY_RESPONSE
    ) is False


def test_is_reliable_outcome_rejects_unknown_outcome() -> None:
    """Tanımsız abstention outcome değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Unknown abstention outcome",
    ):
        is_reliable_outcome(
            "unknown"
        )


def test_classify_both_reliable() -> None:
    """Her iki dil reliable olduğunda both_reliable döndüğünü test eder."""

    assert classify_reliability_transition(
        True,
        True,
    ) == "both_reliable"


def test_classify_source_only_reliable() -> None:
    """Yalnızca source reliable olduğunda source_only_reliable döndüğünü test eder."""

    assert classify_reliability_transition(
        True,
        False,
    ) == "source_only_reliable"


def test_classify_target_only_reliable() -> None:
    """Yalnızca target reliable olduğunda target_only_reliable döndüğünü test eder."""

    assert classify_reliability_transition(
        False,
        True,
    ) == "target_only_reliable"


def test_classify_both_unreliable() -> None:
    """Her iki dil unreliable olduğunda both_unreliable döndüğünü test eder."""

    assert classify_reliability_transition(
        False,
        False,
    ) == "both_unreliable"


def test_group_reliability_by_pair() -> None:
    """Reliability sonuçlarının pair_id bazında gruplandığını test eder."""

    results = [
        make_result(
            "pair_1_en",
            "pair_1",
            "en",
            CORRECT_ANSWER,
            True,
        ),
        make_result(
            "pair_1_az",
            "pair_1",
            "az",
            CORRECT_ANSWER,
            True,
        ),
        make_result(
            "pair_2_en",
            "pair_2",
            "en",
            CORRECT_ABSTENTION,
            False,
        ),
        make_result(
            "pair_2_az",
            "pair_2",
            "az",
            OVER_ANSWERING,
            False,
        ),
    ]

    grouped = group_reliability_by_pair(
        results
    )

    assert set(grouped) == {
        "pair_1",
        "pair_2",
    }

    assert len(grouped["pair_1"]) == 2
    assert len(grouped["pair_2"]) == 2


def test_create_paired_reliability_result() -> None:
    """Geçerli EN-AZ pair'den paired reliability sonucu üretildiğini test eder."""

    pair_results = [
        make_result(
            "pair_1_en",
            "pair_1",
            "en",
            CORRECT_ANSWER,
            True,
        ),
        make_result(
            "pair_1_az",
            "pair_1",
            "az",
            UNDER_ANSWERING,
            True,
        ),
    ]

    result = create_paired_reliability_result(
        pair_results
    )

    assert isinstance(
        result,
        PairedReliabilityResult,
    )

    assert result.pair_id == "pair_1"
    assert result.source_language == "en"
    assert result.target_language == "az"

    assert result.source_outcome == CORRECT_ANSWER
    assert result.target_outcome == UNDER_ANSWERING

    assert result.source_reliable is True
    assert result.target_reliable is False

    assert result.transition == "source_only_reliable"


def test_create_paired_reliability_result_rejects_missing_source() -> None:
    """Source language eksik pair'in reddedildiğini test eder."""

    pair_results = [
        make_result(
            "pair_1_az",
            "pair_1",
            "az",
            CORRECT_ANSWER,
            True,
        )
    ]

    with pytest.raises(
        ValueError,
        match="exactly one 'en' result",
    ):
        create_paired_reliability_result(
            pair_results
        )


def test_create_paired_reliability_result_rejects_missing_target() -> None:
    """Target language eksik pair'in reddedildiğini test eder."""

    pair_results = [
        make_result(
            "pair_1_en",
            "pair_1",
            "en",
            CORRECT_ANSWER,
            True,
        )
    ]

    with pytest.raises(
        ValueError,
        match="exactly one 'az' result",
    ):
        create_paired_reliability_result(
            pair_results
        )


def test_create_paired_reliability_result_rejects_duplicate_language() -> None:
    """Aynı dil için duplicate kayıt bulunan pair'in reddedildiğini test eder."""

    pair_results = [
        make_result(
            "pair_1_en_a",
            "pair_1",
            "en",
            CORRECT_ANSWER,
            True,
        ),
        make_result(
            "pair_1_en_b",
            "pair_1",
            "en",
            CORRECT_ANSWER,
            True,
        ),
        make_result(
            "pair_1_az",
            "pair_1",
            "az",
            CORRECT_ANSWER,
            True,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="exactly one 'en' result",
    ):
        create_paired_reliability_result(
            pair_results
        )


def test_create_paired_reliability_result_rejects_task_mismatch() -> None:
    """Aynı pair içindeki task değerleri farklıysa reddedildiğini test eder."""

    pair_results = [
        make_result(
            "pair_1_en",
            "pair_1",
            "en",
            CORRECT_ANSWER,
            True,
            task="reasoning",
        ),
        make_result(
            "pair_1_az",
            "pair_1",
            "az",
            CORRECT_ANSWER,
            True,
            task="unanswerable",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Task mismatch inside pair",
    ):
        create_paired_reliability_result(
            pair_results
        )


def test_create_paired_reliability_result_rejects_answerability_mismatch() -> None:
    """EN-AZ pair içinde answerability farklıysa reddedildiğini test eder."""

    pair_results = [
        make_result(
            "pair_1_en",
            "pair_1",
            "en",
            CORRECT_ANSWER,
            True,
        ),
        make_result(
            "pair_1_az",
            "pair_1",
            "az",
            CORRECT_ABSTENTION,
            False,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Answerability mismatch inside pair",
    ):
        create_paired_reliability_result(
            pair_results
        )


def test_evaluate_paired_reliability() -> None:
    """Tüm reliability sonuçlarının pair seviyesinde değerlendirildiğini test eder."""

    results = [
        make_result(
            "pair_1_en",
            "pair_1",
            "en",
            CORRECT_ANSWER,
            True,
        ),
        make_result(
            "pair_1_az",
            "pair_1",
            "az",
            UNDER_ANSWERING,
            True,
        ),
        make_result(
            "pair_2_en",
            "pair_2",
            "en",
            CORRECT_ABSTENTION,
            False,
        ),
        make_result(
            "pair_2_az",
            "pair_2",
            "az",
            CORRECT_ABSTENTION,
            False,
        ),
    ]

    paired_results = evaluate_paired_reliability(
        results
    )

    assert len(paired_results) == 2

    assert paired_results[0].pair_id == "pair_1"
    assert paired_results[0].transition == "source_only_reliable"

    assert paired_results[1].pair_id == "pair_2"
    assert paired_results[1].transition == "both_reliable"


def test_evaluate_paired_reliability_preserves_pair_order() -> None:
    """Pair sırasının ilk görülme sırasına göre korunduğunu test eder."""

    results = [
        make_result(
            "pair_2_en",
            "pair_2",
            "en",
            CORRECT_ANSWER,
            True,
        ),
        make_result(
            "pair_2_az",
            "pair_2",
            "az",
            CORRECT_ANSWER,
            True,
        ),
        make_result(
            "pair_1_en",
            "pair_1",
            "en",
            CORRECT_ANSWER,
            True,
        ),
        make_result(
            "pair_1_az",
            "pair_1",
            "az",
            CORRECT_ANSWER,
            True,
        ),
    ]

    paired_results = evaluate_paired_reliability(
        results
    )

    assert [
        result.pair_id
        for result in paired_results
    ] == [
        "pair_2",
        "pair_1",
    ]


def test_paired_reliability_result_to_dict() -> None:
    """PairedReliabilityResult nesnesinin sözlüğe dönüştürüldüğünü test eder."""

    result = PairedReliabilityResult(
        pair_id="pair_1",
        task="reasoning",
        source_language="en",
        target_language="az",
        source_item_id="pair_1_en",
        target_item_id="pair_1_az",
        source_outcome=CORRECT_ANSWER,
        target_outcome=UNDER_ANSWERING,
        source_reliable=True,
        target_reliable=False,
        transition="source_only_reliable",
        metadata={
            "difficulty": "easy",
        },
    )

    data = result.to_dict()

    assert data["pair_id"] == "pair_1"
    assert data["source_reliable"] is True
    assert data["target_reliable"] is False
    assert data["transition"] == "source_only_reliable"


def test_summarize_paired_reliability() -> None:
    """Paired reliability transition sayı ve oranlarının doğru hesaplandığını test eder."""

    results = [
        PairedReliabilityResult(
            pair_id="pair_1",
            task="reasoning",
            source_language="en",
            target_language="az",
            source_item_id="pair_1_en",
            target_item_id="pair_1_az",
            source_outcome=CORRECT_ANSWER,
            target_outcome=CORRECT_ANSWER,
            source_reliable=True,
            target_reliable=True,
            transition="both_reliable",
            metadata={},
        ),
        PairedReliabilityResult(
            pair_id="pair_2",
            task="reasoning",
            source_language="en",
            target_language="az",
            source_item_id="pair_2_en",
            target_item_id="pair_2_az",
            source_outcome=CORRECT_ANSWER,
            target_outcome=UNDER_ANSWERING,
            source_reliable=True,
            target_reliable=False,
            transition="source_only_reliable",
            metadata={},
        ),
        PairedReliabilityResult(
            pair_id="pair_3",
            task="reasoning",
            source_language="en",
            target_language="az",
            source_item_id="pair_3_en",
            target_item_id="pair_3_az",
            source_outcome=OVER_ANSWERING,
            target_outcome=CORRECT_ABSTENTION,
            source_reliable=False,
            target_reliable=True,
            transition="target_only_reliable",
            metadata={},
        ),
        PairedReliabilityResult(
            pair_id="pair_4",
            task="reasoning",
            source_language="en",
            target_language="az",
            source_item_id="pair_4_en",
            target_item_id="pair_4_az",
            source_outcome=OVER_ANSWERING,
            target_outcome=OVER_ANSWERING,
            source_reliable=False,
            target_reliable=False,
            transition="both_unreliable",
            metadata={},
        ),
    ]

    summary = summarize_paired_reliability(
        results
    )

    assert summary["total_pairs"] == 4

    assert summary["both_reliable"] == 1
    assert summary["source_only_reliable"] == 1
    assert summary["target_only_reliable"] == 1
    assert summary["both_unreliable"] == 1

    assert summary["degradation_rate"] == pytest.approx(
        0.25
    )

    assert summary["recovery_rate"] == pytest.approx(
        0.25
    )

    assert summary["consistency_rate"] == pytest.approx(
        0.50
    )


def test_summarize_paired_reliability_rejects_empty_input() -> None:
    """Boş paired reliability listesinin summarize edilmesini reddeder."""

    with pytest.raises(
        ValueError,
        match="Cannot summarize empty paired reliability results",
    ):
        summarize_paired_reliability(
            []
        )


def test_summarize_paired_reliability_rejects_unknown_transition() -> None:
    """Tanımsız reliability transition değerinin reddedildiğini test eder."""

    result = PairedReliabilityResult(
        pair_id="pair_1",
        task="reasoning",
        source_language="en",
        target_language="az",
        source_item_id="pair_1_en",
        target_item_id="pair_1_az",
        source_outcome=CORRECT_ANSWER,
        target_outcome=UNDER_ANSWERING,
        source_reliable=True,
        target_reliable=False,
        transition="unknown",
        metadata={},
    )

    with pytest.raises(
        ValueError,
        match="Unknown reliability transition",
    ):
        summarize_paired_reliability(
            [result]
        )
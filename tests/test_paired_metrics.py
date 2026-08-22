"""Paired EN-AZ capability metric yardımcılarını test eder."""

import pytest

from src.evaluation.exact_match import ExactMatchResult
from src.evaluation.paired_metrics import (
    PairedExactMatchResult,
    classify_transition,
    create_paired_result,
    evaluate_paired_exact_match,
    group_results_by_pair,
    summarize_paired_results,
)


def make_result(
    item_id: str,
    pair_id: str,
    language: str,
    exact_match: int,
    task: str = "reasoning",
) -> ExactMatchResult:
    """Paired metric testleri için örnek ExactMatchResult oluşturur."""

    prediction = "4" if exact_match == 1 else "5"

    return ExactMatchResult(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        prediction=prediction,
        reference_answer="4",
        normalized_prediction=prediction,
        normalized_reference="4",
        exact_match=exact_match,
        metadata={
            "category": "arithmetic_reasoning",
            "difficulty": "easy",
            "review_status": "approved",
        },
    )


def test_classify_both_correct() -> None:
    """Her iki dil doğru olduğunda both_correct döndüğünü test eder."""

    assert classify_transition(
        1,
        1,
    ) == "both_correct"


def test_classify_source_only_correct() -> None:
    """Yalnızca source doğru olduğunda source_only_correct döndüğünü test eder."""

    assert classify_transition(
        1,
        0,
    ) == "source_only_correct"


def test_classify_target_only_correct() -> None:
    """Yalnızca target doğru olduğunda target_only_correct döndüğünü test eder."""

    assert classify_transition(
        0,
        1,
    ) == "target_only_correct"


def test_classify_both_incorrect() -> None:
    """Her iki dil yanlış olduğunda both_incorrect döndüğünü test eder."""

    assert classify_transition(
        0,
        0,
    ) == "both_incorrect"


def test_invalid_source_score_is_rejected() -> None:
    """Geçersiz source exact-match değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Invalid source exact-match value",
    ):
        classify_transition(
            2,
            0,
        )


def test_invalid_target_score_is_rejected() -> None:
    """Geçersiz target exact-match değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Invalid target exact-match value",
    ):
        classify_transition(
            1,
            -1,
        )


def test_group_results_by_pair() -> None:
    """Exact-match sonuçlarının pair_id bazında gruplandığını test eder."""

    results = [
        make_result(
            "reasoning_0001_en",
            "reasoning_0001",
            "en",
            1,
        ),
        make_result(
            "reasoning_0001_az",
            "reasoning_0001",
            "az",
            0,
        ),
        make_result(
            "reasoning_0002_en",
            "reasoning_0002",
            "en",
            1,
        ),
        make_result(
            "reasoning_0002_az",
            "reasoning_0002",
            "az",
            1,
        ),
    ]

    grouped = group_results_by_pair(
        results
    )

    assert set(grouped) == {
        "reasoning_0001",
        "reasoning_0002",
    }

    assert len(grouped["reasoning_0001"]) == 2
    assert len(grouped["reasoning_0002"]) == 2


def test_create_paired_result() -> None:
    """Geçerli EN-AZ pair'den paired result üretildiğini test eder."""

    pair_results = [
        make_result(
            "reasoning_0001_en",
            "reasoning_0001",
            "en",
            1,
        ),
        make_result(
            "reasoning_0001_az",
            "reasoning_0001",
            "az",
            0,
        ),
    ]

    paired = create_paired_result(
        pair_results
    )

    assert isinstance(
        paired,
        PairedExactMatchResult,
    )

    assert paired.pair_id == "reasoning_0001"
    assert paired.source_language == "en"
    assert paired.target_language == "az"
    assert paired.source_exact_match == 1
    assert paired.target_exact_match == 0
    assert paired.transition == "source_only_correct"


def test_create_paired_result_rejects_missing_source() -> None:
    """Source language eksik pair'in reddedildiğini test eder."""

    pair_results = [
        make_result(
            "reasoning_0001_az",
            "reasoning_0001",
            "az",
            1,
        )
    ]

    with pytest.raises(
        ValueError,
        match="exactly one 'en' result",
    ):
        create_paired_result(
            pair_results
        )


def test_create_paired_result_rejects_missing_target() -> None:
    """Target language eksik pair'in reddedildiğini test eder."""

    pair_results = [
        make_result(
            "reasoning_0001_en",
            "reasoning_0001",
            "en",
            1,
        )
    ]

    with pytest.raises(
        ValueError,
        match="exactly one 'az' result",
    ):
        create_paired_result(
            pair_results
        )


def test_create_paired_result_rejects_duplicate_language() -> None:
    """Aynı dil için duplicate kayıt bulunan pair'in reddedildiğini test eder."""

    pair_results = [
        make_result(
            "reasoning_0001_en_a",
            "reasoning_0001",
            "en",
            1,
        ),
        make_result(
            "reasoning_0001_en_b",
            "reasoning_0001",
            "en",
            1,
        ),
        make_result(
            "reasoning_0001_az",
            "reasoning_0001",
            "az",
            1,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="exactly one 'en' result",
    ):
        create_paired_result(
            pair_results
        )


def test_create_paired_result_rejects_task_mismatch() -> None:
    """Aynı pair içindeki task değerleri farklıysa reddedildiğini test eder."""

    pair_results = [
        make_result(
            "reasoning_0001_en",
            "reasoning_0001",
            "en",
            1,
            task="reasoning",
        ),
        make_result(
            "reasoning_0001_az",
            "reasoning_0001",
            "az",
            1,
            task="factual_knowledge",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Task mismatch inside pair",
    ):
        create_paired_result(
            pair_results
        )


def test_evaluate_paired_exact_match() -> None:
    """Tüm exact-match sonuçlarının pair seviyesinde değerlendirildiğini test eder."""

    results = [
        make_result(
            "reasoning_0001_en",
            "reasoning_0001",
            "en",
            1,
        ),
        make_result(
            "reasoning_0001_az",
            "reasoning_0001",
            "az",
            0,
        ),
        make_result(
            "reasoning_0002_en",
            "reasoning_0002",
            "en",
            1,
        ),
        make_result(
            "reasoning_0002_az",
            "reasoning_0002",
            "az",
            1,
        ),
    ]

    paired_results = evaluate_paired_exact_match(
        results
    )

    assert len(paired_results) == 2

    assert paired_results[0].pair_id == "reasoning_0001"
    assert paired_results[0].transition == "source_only_correct"

    assert paired_results[1].pair_id == "reasoning_0002"
    assert paired_results[1].transition == "both_correct"


def test_paired_result_to_dict() -> None:
    """PairedExactMatchResult nesnesinin sözlüğe dönüştürüldüğünü test eder."""

    result = PairedExactMatchResult(
        pair_id="reasoning_0001",
        task="reasoning",
        source_language="en",
        target_language="az",
        source_item_id="reasoning_0001_en",
        target_item_id="reasoning_0001_az",
        source_exact_match=1,
        target_exact_match=0,
        transition="source_only_correct",
        metadata={
            "category": "arithmetic_reasoning",
        },
    )

    assert result.to_dict()["transition"] == "source_only_correct"
    assert result.to_dict()["pair_id"] == "reasoning_0001"


def test_summarize_paired_results() -> None:
    """Paired transition dağılımının ve oranlarının doğru hesaplandığını test eder."""

    results = [
        PairedExactMatchResult(
            pair_id="pair_1",
            task="reasoning",
            source_language="en",
            target_language="az",
            source_item_id="pair_1_en",
            target_item_id="pair_1_az",
            source_exact_match=1,
            target_exact_match=1,
            transition="both_correct",
            metadata={},
        ),
        PairedExactMatchResult(
            pair_id="pair_2",
            task="reasoning",
            source_language="en",
            target_language="az",
            source_item_id="pair_2_en",
            target_item_id="pair_2_az",
            source_exact_match=1,
            target_exact_match=0,
            transition="source_only_correct",
            metadata={},
        ),
        PairedExactMatchResult(
            pair_id="pair_3",
            task="reasoning",
            source_language="en",
            target_language="az",
            source_item_id="pair_3_en",
            target_item_id="pair_3_az",
            source_exact_match=0,
            target_exact_match=1,
            transition="target_only_correct",
            metadata={},
        ),
        PairedExactMatchResult(
            pair_id="pair_4",
            task="reasoning",
            source_language="en",
            target_language="az",
            source_item_id="pair_4_en",
            target_item_id="pair_4_az",
            source_exact_match=0,
            target_exact_match=0,
            transition="both_incorrect",
            metadata={},
        ),
    ]

    summary = summarize_paired_results(
        results
    )

    assert summary["total_pairs"] == 4
    assert summary["both_correct"] == 1
    assert summary["source_only_correct"] == 1
    assert summary["target_only_correct"] == 1
    assert summary["both_incorrect"] == 1

    assert summary["degradation_rate"] == pytest.approx(
        0.25
    )

    assert summary["recovery_rate"] == pytest.approx(
        0.25
    )

    assert summary["consistency_rate"] == pytest.approx(
        0.50
    )


def test_summarize_paired_results_rejects_empty_input() -> None:
    """Boş paired result listesinin summarize edilmesinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Cannot summarize empty paired results",
    ):
        summarize_paired_results(
            []
        )


def test_summarize_paired_results_rejects_unknown_transition() -> None:
    """Tanımsız transition değerinin aggregate analizde reddedildiğini test eder."""

    result = PairedExactMatchResult(
        pair_id="pair_1",
        task="reasoning",
        source_language="en",
        target_language="az",
        source_item_id="pair_1_en",
        target_item_id="pair_1_az",
        source_exact_match=1,
        target_exact_match=0,
        transition="unknown",
        metadata={},
    )

    with pytest.raises(
        ValueError,
        match="Unknown paired transition",
    ):
        summarize_paired_results(
            [result]
        )
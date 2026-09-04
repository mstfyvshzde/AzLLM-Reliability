"""Task-aware capability evaluator testleri."""

from src.evaluation.run_inference import PredictionRecord
from src.evaluation.task_aware_capability import (
    evaluate_task_aware_prediction,
)
from src.evaluation.task_aware_capability import (
    evaluate_task_aware_capability,
)


def make_record(
    task: str,
    prediction: str,
    reference_answer: str,
    category: str = "test",
) -> PredictionRecord:
    return PredictionRecord(
        item_id=f"{task}_001_en",
        pair_id=f"{task}_001",
        language="en",
        task=task,
        question="Test question",
        reference_answer=reference_answer,
        prediction=prediction,
        metadata={
            "category": category,
            "difficulty": "easy",
            "is_answerable": True,
        },
    )


def test_reasoning_uses_short_answer() -> None:
    record = make_record(
        task="reasoning",
        prediction="Nigar is the shortest.",
        reference_answer="Nigar",
    )

    result = evaluate_task_aware_prediction(record)

    assert result is not None
    assert result.evaluator == "short_answer"
    assert result.correct == 1


def test_factual_knowledge_uses_short_answer() -> None:
    record = make_record(
        task="factual_knowledge",
        prediction="The capital is Tokyo.",
        reference_answer="Tokyo",
    )

    result = evaluate_task_aware_prediction(record)

    assert result is not None
    assert result.evaluator == "short_answer"
    assert result.correct == 1


def test_linguistic_understanding_uses_semantic_adjudication() -> None:
    record = make_record(
        task="linguistic_understanding",
        prediction="It means explain in more detail.",
        reference_answer="Explain in greater detail",
        category="contextual_meaning",
    )

    result = evaluate_task_aware_prediction(
        record,
        semantic_adjudication_decisions={
            record.item_id: 1,
        },
    )

    assert result is not None
    assert result.evaluator == "semantic_adjudication"
    assert result.correct == 1


def test_semantic_adjudication_decision_is_required() -> None:
    import pytest

    record = make_record(
        task="linguistic_understanding",
        prediction="Some answer.",
        reference_answer="Some meaning",
        category="contextual_meaning",
    )

    with pytest.raises(
        ValueError,
        match="Semantic adjudication decisions are required",
    ):
        evaluate_task_aware_prediction(record)


def test_missing_semantic_adjudication_item_is_rejected() -> None:
    import pytest

    record = make_record(
        task="linguistic_understanding",
        prediction="Some answer.",
        reference_answer="Some meaning",
        category="lexical_disambiguation",
    )

    with pytest.raises(
        ValueError,
        match="Missing semantic adjudication decision",
    ):
        evaluate_task_aware_prediction(
            record,
            semantic_adjudication_decisions={},
        )


def test_unknown_linguistic_category_is_rejected() -> None:
    import pytest

    record = make_record(
        task="linguistic_understanding",
        prediction="Some answer.",
        reference_answer="Some meaning",
        category="unknown_category",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported linguistic_understanding category",
    ):
        evaluate_task_aware_prediction(record)


def test_instruction_following_uses_exact_match() -> None:
    record = make_record(
        task="instruction_following",
        prediction="RESULT: red",
        reference_answer="RESULT: red",
    )

    result = evaluate_task_aware_prediction(record)

    assert result is not None
    assert result.evaluator == "instruction_following"
    assert result.correct == 1


def test_unanswerable_is_excluded() -> None:
    record = make_record(
        task="unanswerable",
        prediction="Cannot be determined.",
        reference_answer="Cannot be determined.",
    )

    result = evaluate_task_aware_prediction(record)

    assert result is None




def test_evaluate_task_aware_capability() -> None:
    records = [
        make_record(
            task="reasoning",
            prediction="Nigar is the shortest.",
            reference_answer="Nigar",
        ),
        make_record(
            task="unanswerable",
            prediction="Cannot be determined.",
            reference_answer="Cannot be determined.",
        ),
    ]

    results = evaluate_task_aware_capability(
        records
    )

    assert len(results) == 1
    assert results[0].task == "reasoning"
    assert results[0].correct == 1


from src.evaluation.task_aware_capability import (
    calculate_task_aware_accuracy,
    summarize_task_aware_capability,
)


def test_calculate_task_aware_accuracy() -> None:
    records = [
        make_record(
            task="reasoning",
            prediction="Nigar is the shortest.",
            reference_answer="Nigar",
        ),
        make_record(
            task="reasoning",
            prediction="Aysel is the shortest.",
            reference_answer="Nigar",
        ),
    ]

    results = evaluate_task_aware_capability(
        records
    )

    assert calculate_task_aware_accuracy(
        results
    ) == 0.5


def test_summarize_task_aware_capability() -> None:
    records = [
        make_record(
            task="reasoning",
            prediction="Nigar is the shortest.",
            reference_answer="Nigar",
        ),
        make_record(
            task="reasoning",
            prediction="Aysel is the shortest.",
            reference_answer="Nigar",
        ),
        make_record(
            task="instruction_following",
            prediction="RESULT: red",
            reference_answer="RESULT: red",
        ),
    ]

    results = evaluate_task_aware_capability(
        records
    )

    summary = summarize_task_aware_capability(
        results
    )

    assert summary["overall"] == {
        "total": 3,
        "correct": 2,
        "incorrect": 1,
        "accuracy": 2 / 3,
    }

    assert summary["by_task"]["reasoning"] == {
        "total": 2,
        "correct": 1,
        "incorrect": 1,
        "accuracy": 0.5,
    }

    assert summary["by_task"]["instruction_following"] == {
        "total": 1,
        "correct": 1,
        "incorrect": 0,
        "accuracy": 1.0,
    }


def test_binary_reasoning_uses_binary_answer() -> None:
    record = make_record(
        task="reasoning",
        prediction="Yes, Q occurred.",
        reference_answer="Yes",
        category="logical_reasoning",
    )

    result = evaluate_task_aware_prediction(
        record
    )

    assert result is not None
    assert result.evaluator == "binary_answer"
    assert result.correct == 1


def test_indirect_binary_reasoning_uses_adjudication() -> None:
    record = make_record(
        task="reasoning",
        prediction="Q must have occurred.",
        reference_answer="Yes",
        category="logical_reasoning",
    )

    result = evaluate_task_aware_prediction(
        record,
        binary_adjudication_decisions={
            record.item_id: 1,
        },
    )

    assert result is not None
    assert result.evaluator == "binary_answer"
    assert result.correct == 1


def test_indirect_binary_reasoning_requires_adjudication() -> None:
    import pytest

    record = make_record(
        task="reasoning",
        prediction="Q must have occurred.",
        reference_answer="Yes",
        category="logical_reasoning",
    )

    with pytest.raises(
        ValueError,
        match="Binary adjudication decisions are required",
    ):
        evaluate_task_aware_prediction(
            record
        )


def test_non_binary_reasoning_still_uses_short_answer() -> None:
    record = make_record(
        task="reasoning",
        prediction="Nigar is the shortest.",
        reference_answer="Nigar",
        category="comparative_reasoning",
    )

    result = evaluate_task_aware_prediction(
        record
    )

    assert result is not None
    assert result.evaluator == "short_answer"
    assert result.correct == 1

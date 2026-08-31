"""End-to-end evaluation pipeline yardımcılarını test eder."""

import json
from pathlib import Path

import pytest

from src.evaluation.evaluate_predictions import (
    build_capability_summary,
    build_reliability_summary,
    evaluate_predictions,
    load_predictions,
    save_evaluation_artifacts,
    save_json,
    save_jsonl,
)
from src.evaluation.exact_match import (
    ExactMatchResult,
    evaluate_exact_match,
)
from src.evaluation.run_inference import PredictionRecord
from src.reliability.abstention_metrics import (
    AbstentionResult,
    evaluate_abstention,
)


def make_prediction(
    item_id: str,
    pair_id: str,
    language: str,
    prediction: str,
    reference_answer: str,
    is_answerable: bool,
    task: str = "reasoning",
) -> PredictionRecord:
    """Evaluation pipeline testleri için örnek PredictionRecord oluşturur."""

    return PredictionRecord(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        question="What is the answer?",
        reference_answer=reference_answer,
        prediction=prediction,
        metadata={
            "category": "arithmetic_reasoning",
            "difficulty": "easy",
            "review_status": "approved",
            "is_answerable": is_answerable,
        },
    )


def make_paired_predictions() -> list[PredictionRecord]:
    """EN-AZ paired capability ve reliability test datası oluşturur."""

    return [
        make_prediction(
            item_id="pair_1_en",
            pair_id="pair_1",
            language="en",
            prediction="4",
            reference_answer="4",
            is_answerable=True,
        ),
        make_prediction(
            item_id="pair_1_az",
            pair_id="pair_1",
            language="az",
            prediction="5",
            reference_answer="4",
            is_answerable=True,
        ),
        make_prediction(
            item_id="pair_2_en",
            pair_id="pair_2",
            language="en",
            prediction="I don't know.",
            reference_answer="",
            is_answerable=False,
        ),
        make_prediction(
            item_id="pair_2_az",
            pair_id="pair_2",
            language="az",
            prediction="42",
            reference_answer="",
            is_answerable=False,
        ),
    ]


def test_load_predictions(tmp_path: Path) -> None:
    """Prediction JSONL dosyasının doğru yüklendiğini test eder."""

    input_path = tmp_path / "predictions.jsonl"

    records = make_paired_predictions()

    with input_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            json.dump(
                record.to_dict(),
                file,
                ensure_ascii=False,
            )
            file.write("\n")

    loaded = load_predictions(
        input_path
    )

    assert len(loaded) == 4
    assert loaded[0].item_id == "pair_1_en"
    assert loaded[1].language == "az"
    assert loaded[2].metadata["is_answerable"] is False


def test_load_predictions_ignores_empty_lines(
    tmp_path: Path,
) -> None:
    """Prediction dosyasındaki boş satırların ignore edildiğini test eder."""

    input_path = tmp_path / "predictions.jsonl"

    record = make_prediction(
        item_id="item_1",
        pair_id="pair_1",
        language="en",
        prediction="4",
        reference_answer="4",
        is_answerable=True,
    )

    with input_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n")

        json.dump(
            record.to_dict(),
            file,
        )

        file.write("\n\n")

    loaded = load_predictions(
        input_path
    )

    assert len(loaded) == 1


def test_load_predictions_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Mevcut olmayan prediction dosyasının reddedildiğini test eder."""

    input_path = tmp_path / "missing.jsonl"

    with pytest.raises(
        FileNotFoundError,
        match="Prediction file not found",
    ):
        load_predictions(
            input_path
        )


def test_load_predictions_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Geçersiz JSON satırının reddedildiğini test eder."""

    input_path = tmp_path / "predictions.jsonl"

    input_path.write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON at line 1",
    ):
        load_predictions(
            input_path
        )


def test_load_predictions_rejects_missing_required_field(
    tmp_path: Path,
) -> None:
    """Required prediction alanı eksikse hata üretildiğini test eder."""

    input_path = tmp_path / "predictions.jsonl"

    data = {
        "item_id": "item_1",
        "pair_id": "pair_1",
        "language": "en",
        "task": "reasoning",
        "question": "Question?",
        "reference_answer": "4",
        "metadata": {
            "is_answerable": True,
        },
    }

    input_path.write_text(
        json.dumps(data) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing prediction field 'prediction'",
    ):
        load_predictions(
            input_path
        )


def test_save_json(
    tmp_path: Path,
) -> None:
    """JSON summary artifact'ının kaydedildiğini test eder."""

    output_path = tmp_path / "summary.json"

    save_json(
        {
            "accuracy": 0.5,
        },
        output_path,
    )

    saved = json.loads(
        output_path.read_text(
            encoding="utf-8",
        )
    )

    assert saved == {
        "accuracy": 0.5,
    }


def test_save_json_rejects_existing_file(
    tmp_path: Path,
) -> None:
    """Overwrite kapalıyken mevcut JSON dosyasının korunmasını test eder."""

    output_path = tmp_path / "summary.json"

    output_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="Output file already exists",
    ):
        save_json(
            {
                "accuracy": 1.0,
            },
            output_path,
        )


def test_save_json_overwrites_existing_file(
    tmp_path: Path,
) -> None:
    """Overwrite açıkken mevcut JSON artifact'ın güncellendiğini test eder."""

    output_path = tmp_path / "summary.json"

    output_path.write_text(
        "{}",
        encoding="utf-8",
    )

    save_json(
        {
            "accuracy": 1.0,
        },
        output_path,
        overwrite=True,
    )

    saved = json.loads(
        output_path.read_text(
            encoding="utf-8",
        )
    )

    assert saved["accuracy"] == 1.0


def test_save_jsonl(
    tmp_path: Path,
) -> None:
    """Item-level evaluation kayıtlarının JSONL olarak kaydedildiğini test eder."""

    predictions = make_paired_predictions()

    results = evaluate_exact_match(
        predictions
    )

    output_path = tmp_path / "results.jsonl"

    save_jsonl(
        results,
        output_path,
    )

    lines = output_path.read_text(
        encoding="utf-8",
    ).strip().splitlines()

    assert len(lines) == 4

    first = json.loads(
        lines[0]
    )

    assert first["item_id"] == "pair_1_en"
    assert first["exact_match"] == 1


def test_build_capability_summary() -> None:
    """Capability summary'nin expected bölümleri içerdiğini test eder."""

    predictions = make_paired_predictions()

    results = evaluate_exact_match(
        predictions
    )

    summary = build_capability_summary(
        results
    )

    assert "overall" in summary
    assert "by_language" in summary
    assert "by_task" in summary
    assert "by_category" in summary
    assert "by_difficulty" in summary
    assert "language_task_matrix" in summary
    assert "language_gap" in summary

    assert summary["by_language"]["en"]["total"] == 2
    assert summary["by_language"]["az"]["total"] == 2


def test_build_capability_summary_rejects_empty_results() -> None:
    """Boş capability sonuçlarının summarize edilmesini reddeder."""

    with pytest.raises(
        ValueError,
        match="Cannot build capability summary from empty results",
    ):
        build_capability_summary(
            []
        )


def test_build_reliability_summary() -> None:
    """Reliability summary'nin expected metric bölümlerini içerdiğini test eder."""

    predictions = make_paired_predictions()

    results = evaluate_abstention(
        predictions
    )

    summary = build_reliability_summary(
        results
    )

    assert "overall" in summary
    assert "by_language" in summary
    assert "by_task" in summary
    assert "by_category" in summary
    assert "by_difficulty" in summary
    assert "language_gap" in summary

    assert summary["by_language"]["en"]["total"] == 2
    assert summary["by_language"]["az"]["total"] == 2


def test_build_reliability_summary_rejects_empty_results() -> None:
    """Boş reliability sonuçlarının summarize edilmesini reddeder."""

    with pytest.raises(
        ValueError,
        match="Cannot build reliability summary from empty results",
    ):
        build_reliability_summary(
            []
        )


def test_evaluate_predictions() -> None:
    """Tüm evaluation pipeline'ın tek çağrıda çalıştığını test eder."""

    predictions = make_paired_predictions()

    evaluation = evaluate_predictions(
        predictions
    )

    assert len(
        evaluation["exact_match_results"]
    ) == 4

    assert len(
        evaluation["paired_capability_results"]
    ) == 2

    assert len(
        evaluation["abstention_results"]
    ) == 4

    assert len(
        evaluation["paired_reliability_results"]
    ) == 2

    assert "capability_summary" in evaluation
    assert "paired_capability_summary" in evaluation
    assert "reliability_summary" in evaluation
    assert "paired_reliability_summary" in evaluation


def test_evaluate_predictions_rejects_empty_input() -> None:
    """Boş prediction listesinin evaluation pipeline'a verilmesini reddeder."""

    with pytest.raises(
        ValueError,
        match="Cannot evaluate empty prediction records",
    ):
        evaluate_predictions(
            []
        )


def test_evaluate_predictions_capability_behavior() -> None:
    """Paired capability transition'larının beklenen davranışı verdiğini test eder."""

    evaluation = evaluate_predictions(
        make_paired_predictions()
    )

    summary = evaluation[
        "paired_capability_summary"
    ]

    assert summary["total_pairs"] == 2

    assert summary[
        "source_only_correct"
    ] == 1


def test_evaluate_predictions_reliability_behavior() -> None:
    """Paired reliability transition'larının expected sonucu verdiğini test eder."""

    evaluation = evaluate_predictions(
        make_paired_predictions()
    )

    summary = evaluation[
        "paired_reliability_summary"
    ]

    assert summary["total_pairs"] == 2

    assert summary[
        "both_reliable"
    ] == 1

    assert summary[
        "source_only_reliable"
    ] == 1


def test_save_evaluation_artifacts(
    tmp_path: Path,
) -> None:
    """Tüm evaluation artifact dosyalarının kaydedildiğini test eder."""

    evaluation = evaluate_predictions(
        make_paired_predictions()
    )

    output_dir = tmp_path / "evaluation"

    save_evaluation_artifacts(
        evaluation,
        output_dir,
    )

    expected_files = {
        "exact_match_results.jsonl",
        "paired_capability_results.jsonl",
        "abstention_results.jsonl",
        "paired_reliability_results.jsonl",
        "capability_summary.json",
        "paired_capability_summary.json",
        "reliability_summary.json",
        "paired_reliability_summary.json",
        "short_answer_results.jsonl",
        "short_answer_summary.json",
        "semantic_answer_results.jsonl",
        "semantic_answer_summary.json",
        "task_aware_results.jsonl", 
        "task_aware_summary.json",
        "paired_task_aware_results.jsonl",
        "paired_task_aware_summary.json",
        "primary_capability_summary.json",
        "primary_paired_capability_summary.json",
        "instruction_following_results.jsonl",
        "instruction_following_summary.json",
    }

    actual_files = {
        path.name
        for path in output_dir.iterdir()
    }

    assert actual_files == expected_files


def test_save_evaluation_artifacts_rejects_existing_files(
    tmp_path: Path,
) -> None:
    """Overwrite kapalıyken existing artifact'ların korunmasını test eder."""

    evaluation = evaluate_predictions(
        make_paired_predictions()
    )

    output_dir = tmp_path / "evaluation"

    save_evaluation_artifacts(
        evaluation,
        output_dir,
    )

    with pytest.raises(
        FileExistsError,
        match="Output file already exists",
    ):
        save_evaluation_artifacts(
            evaluation,
            output_dir,
        )


def test_save_evaluation_artifacts_overwrites_existing_files(
    tmp_path: Path,
) -> None:
    """Overwrite açıkken evaluation artifact'larının tekrar yazıldığını test eder."""

    evaluation = evaluate_predictions(
        make_paired_predictions()
    )

    output_dir = tmp_path / "evaluation"

    save_evaluation_artifacts(
        evaluation,
        output_dir,
    )

    save_evaluation_artifacts(
        evaluation,
        output_dir,
        overwrite=True,
    )

    assert (
        output_dir
        / "capability_summary.json"
    ).exists()



def test_evaluate_predictions_includes_short_answer_summary() -> None:
    evaluation = evaluate_predictions(
        make_paired_predictions()
    )

    assert "short_answer_results" in evaluation
    assert "short_answer_summary" in evaluation

    summary = evaluation[
        "short_answer_summary"
    ]

    assert summary["total"] > 0
    assert 0.0 <= summary["accuracy"] <= 1.0
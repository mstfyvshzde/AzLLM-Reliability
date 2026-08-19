"""Benchmark inference pipeline davranışını test eder."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.benchmark_record import BenchmarkRecord
from src.evaluation.run_inference import (
    PredictionRecord,
    create_prediction_record,
    run_inference,
    save_predictions,
)
from src.models.model_config import ModelConfig


def make_record(
    item_id: str = "reasoning_0001_en",
    pair_id: str = "reasoning_0001",
    language: str = "en",
    question: str = "What is 2 + 2?",
    reference_answer: str = "4",
) -> BenchmarkRecord:
    """Inference testleri için örnek BenchmarkRecord oluşturur."""

    return BenchmarkRecord(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task="reasoning",
        question=question,
        reference_answer=reference_answer,
        metadata={
            "category": "arithmetic_reasoning",
            "difficulty": "easy",
            "review_status": "approved",
        },
    )


def make_config() -> ModelConfig:
    """Inference testlerinde kullanılacak minimal ModelConfig oluşturur."""

    return ModelConfig(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        device="cpu",
        dtype="float32",
        max_new_tokens=64,
        temperature=0.0,
        do_sample=False,
    )


def test_prediction_record_to_dict() -> None:
    """PredictionRecord nesnesinin sözlüğe doğru dönüştürüldüğünü test eder."""

    prediction = PredictionRecord(
        item_id="reasoning_0001_en",
        pair_id="reasoning_0001",
        language="en",
        task="reasoning",
        question="What is 2 + 2?",
        reference_answer="4",
        prediction="4",
        metadata={
            "category": "arithmetic_reasoning",
        },
    )

    result = prediction.to_dict()

    assert result == {
        "item_id": "reasoning_0001_en",
        "pair_id": "reasoning_0001",
        "language": "en",
        "task": "reasoning",
        "question": "What is 2 + 2?",
        "reference_answer": "4",
        "prediction": "4",
        "metadata": {
            "category": "arithmetic_reasoning",
        },
    }


def test_create_prediction_record() -> None:
    """BenchmarkRecord ve prediction değerinden doğru PredictionRecord üretildiğini test eder."""

    record = make_record()

    prediction_record = create_prediction_record(
        record=record,
        prediction="4",
    )

    assert prediction_record.item_id == record.item_id
    assert prediction_record.pair_id == record.pair_id
    assert prediction_record.language == record.language
    assert prediction_record.task == record.task
    assert prediction_record.question == record.question
    assert prediction_record.reference_answer == record.reference_answer
    assert prediction_record.prediction == "4"
    assert prediction_record.metadata == record.metadata


def test_create_prediction_record_preserves_empty_prediction() -> None:
    """Boş model cevabının PredictionRecord içinde korunabildiğini test eder."""

    record = make_record()

    prediction_record = create_prediction_record(
        record=record,
        prediction="",
    )

    assert prediction_record.prediction == ""


@patch("src.evaluation.run_inference.generate_response")
def test_run_inference(
    mock_generate_response,
) -> None:
    """Her benchmark kaydı için generation çalıştırıldığını test eder."""

    records = [
        make_record(
            item_id="reasoning_0001_en",
            pair_id="reasoning_0001",
            language="en",
            question="What is 2 + 2?",
        ),
        make_record(
            item_id="reasoning_0001_az",
            pair_id="reasoning_0001",
            language="az",
            question="2 + 2 neçə edir?",
        ),
    ]

    model = MagicMock()
    tokenizer = MagicMock()
    config = make_config()

    mock_generate_response.side_effect = [
        "4",
        "4",
    ]

    predictions = run_inference(
        records=records,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    assert len(predictions) == 2

    assert predictions[0].item_id == "reasoning_0001_en"
    assert predictions[0].prediction == "4"

    assert predictions[1].item_id == "reasoning_0001_az"
    assert predictions[1].prediction == "4"

    assert mock_generate_response.call_count == 2


@patch("src.evaluation.run_inference.generate_response")
def test_run_inference_preserves_input_order(
    mock_generate_response,
) -> None:
    """Prediction çıktısında benchmark input sırasının korunduğunu test eder."""

    records = [
        make_record(
            item_id="reasoning_0002_az",
            pair_id="reasoning_0002",
            language="az",
            question="3 + 3 neçə edir?",
            reference_answer="6",
        ),
        make_record(
            item_id="reasoning_0001_en",
            pair_id="reasoning_0001",
            language="en",
            question="What is 2 + 2?",
            reference_answer="4",
        ),
    ]

    model = MagicMock()
    tokenizer = MagicMock()
    config = make_config()

    mock_generate_response.side_effect = [
        "6",
        "4",
    ]

    predictions = run_inference(
        records=records,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    assert [
        prediction.item_id
        for prediction in predictions
    ] == [
        "reasoning_0002_az",
        "reasoning_0001_en",
    ]


@patch("src.evaluation.run_inference.generate_response")
def test_run_inference_uses_same_model_config(
    mock_generate_response,
) -> None:
    """Tüm benchmark kayıtlarında aynı model, tokenizer ve config kullanıldığını test eder."""

    records = [
        make_record(
            item_id="reasoning_0001_en",
            pair_id="reasoning_0001",
            language="en",
        ),
        make_record(
            item_id="reasoning_0001_az",
            pair_id="reasoning_0001",
            language="az",
        ),
    ]

    model = MagicMock()
    tokenizer = MagicMock()
    config = make_config()

    mock_generate_response.return_value = "4"

    run_inference(
        records=records,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    for call in mock_generate_response.call_args_list:
        assert call.kwargs["model"] is model
        assert call.kwargs["tokenizer"] is tokenizer
        assert call.kwargs["config"] is config


def test_save_predictions(tmp_path: Path) -> None:
    """Prediction kayıtlarının JSONL dosyasına doğru yazıldığını test eder."""

    predictions = [
        PredictionRecord(
            item_id="reasoning_0001_en",
            pair_id="reasoning_0001",
            language="en",
            task="reasoning",
            question="What is 2 + 2?",
            reference_answer="4",
            prediction="4",
            metadata={
                "category": "arithmetic_reasoning",
            },
        ),
        PredictionRecord(
            item_id="reasoning_0001_az",
            pair_id="reasoning_0001",
            language="az",
            task="reasoning",
            question="2 + 2 neçə edir?",
            reference_answer="4",
            prediction="4",
            metadata={
                "category": "arithmetic_reasoning",
            },
        ),
    ]

    output_path = tmp_path / "predictions.jsonl"

    save_predictions(
        predictions=predictions,
        output_path=output_path,
    )

    assert output_path.exists()

    lines = output_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert len(lines) == 2

    english_record = json.loads(lines[0])
    azerbaijani_record = json.loads(lines[1])

    assert english_record["item_id"] == "reasoning_0001_en"
    assert english_record["prediction"] == "4"

    assert azerbaijani_record["item_id"] == "reasoning_0001_az"
    assert azerbaijani_record["prediction"] == "4"


def test_save_predictions_rejects_existing_output(
    tmp_path: Path,
) -> None:
    """Overwrite kapalıyken mevcut prediction dosyasının üzerine yazılmadığını test eder."""

    output_path = tmp_path / "predictions.jsonl"

    output_path.write_text(
        "existing",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="Prediction output already exists",
    ):
        save_predictions(
            predictions=[],
            output_path=output_path,
            overwrite=False,
        )


def test_save_predictions_allows_overwrite(
    tmp_path: Path,
) -> None:
    """Overwrite açıkken mevcut prediction dosyasının üzerine yazılabildiğini test eder."""

    output_path = tmp_path / "predictions.jsonl"

    output_path.write_text(
        "existing",
        encoding="utf-8",
    )

    predictions = [
        PredictionRecord(
            item_id="reasoning_0001_en",
            pair_id="reasoning_0001",
            language="en",
            task="reasoning",
            question="What is 2 + 2?",
            reference_answer="4",
            prediction="4",
            metadata={},
        )
    ]

    save_predictions(
        predictions=predictions,
        output_path=output_path,
        overwrite=True,
    )

    content = output_path.read_text(
        encoding="utf-8"
    )

    assert "reasoning_0001_en" in content
    assert '"prediction": "4"' in content
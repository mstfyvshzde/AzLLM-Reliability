"""Baseline preflight yardımcı fonksiyonlarını test eder."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.benchmark_record import BenchmarkRecord
from src.experiments.preflight import (
    PreflightResult,
    run_preflight,
    validate_benchmark_pairs,
    validate_existing_output,
    validate_output_paths,
)


def make_record(
    item_id: str,
    pair_id: str,
    language: str,
    task: str = "reasoning",
) -> BenchmarkRecord:
    """Preflight testleri için örnek BenchmarkRecord oluşturur."""

    return BenchmarkRecord(
        item_id=item_id,
        pair_id=pair_id,
        language=language,
        task=task,
        question="What is the answer?",
        reference_answer="4",
        metadata={
            "category": "arithmetic_reasoning",
            "difficulty": "easy",
            "review_status": "approved",
            "is_answerable": True,
        },
    )


def make_config(
    tmp_path: Path,
) -> dict:
    """Preflight testleri için minimal geçerli experiment config oluşturur."""

    return {
        "experiment": {
            "name": "baseline",
            "seed": 17,
        },
        "languages": {
            "source": "en",
            "target": "az",
        },
        "model": {
            "config_path": str(
                tmp_path
                / "base.yaml"
            ),
        },
        "benchmark": {
            "split": "test",
            "input_path": str(
                tmp_path
                / "test.jsonl"
            ),
            "require_complete_pairs": True,
        },
        "generation": {
            "use_model_config": True,
        },
        "inference": {
            "output_path": str(
                tmp_path
                / "outputs"
                / "predictions.jsonl"
            ),
            "overwrite": False,
        },
        "evaluation": {
            "output_dir": str(
                tmp_path
                / "outputs"
                / "evaluation"
            ),
            "overwrite": False,
        },
        "artifacts": {
            "save_config_snapshot": True,
            "save_run_metadata": True,
        },
    }


def test_preflight_result_to_dict() -> None:
    """PreflightResult nesnesinin sözlüğe doğru dönüştürüldüğünü test eder."""

    result = PreflightResult(
        experiment_name="baseline",
        benchmark_record_count=4,
        pair_count=2,
        source_language="en",
        target_language="az",
        model_name="test-model",
        ready=True,
    )

    assert result.to_dict() == {
        "experiment_name": "baseline",
        "benchmark_record_count": 4,
        "pair_count": 2,
        "source_language": "en",
        "target_language": "az",
        "model_name": "test-model",
        "ready": True,
    }


def test_validate_existing_output_allows_missing_file(
    tmp_path: Path,
) -> None:
    """Mevcut olmayan output path'in geçerli olduğunu test eder."""

    validate_existing_output(
        output_path=tmp_path / "missing.jsonl",
        overwrite=False,
        artifact_name="Prediction artifact",
    )


def test_validate_existing_output_rejects_existing_file(
    tmp_path: Path,
) -> None:
    """Overwrite kapalıyken mevcut artifact'ın reddedildiğini test eder."""

    output_path = (
        tmp_path
        / "predictions.jsonl"
    )

    output_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="Prediction artifact already exists",
    ):
        validate_existing_output(
            output_path=output_path,
            overwrite=False,
            artifact_name="Prediction artifact",
        )


def test_validate_existing_output_allows_overwrite(
    tmp_path: Path,
) -> None:
    """Overwrite açıkken mevcut artifact'ın kabul edildiğini test eder."""

    output_path = (
        tmp_path
        / "predictions.jsonl"
    )

    output_path.write_text(
        "{}",
        encoding="utf-8",
    )

    validate_existing_output(
        output_path=output_path,
        overwrite=True,
        artifact_name="Prediction artifact",
    )


def test_validate_output_paths() -> None:
    """Boş output path'lerin preflight kontrolünden geçtiğini test eder."""

    config = {
        "inference": {
            "output_path": "outputs/test/predictions.jsonl",
            "overwrite": False,
        },
        "evaluation": {
            "output_dir": "outputs/test/evaluation",
            "overwrite": False,
        },
        "artifacts": {
            "save_config_snapshot": False,
            "save_run_metadata": False,
        },
    }

    validate_output_paths(
        config
    )


def test_validate_output_paths_rejects_prediction_artifact(
    tmp_path: Path,
) -> None:
    """Existing prediction artifact'ın overwrite kapalıyken reddedildiğini test eder."""

    prediction_path = (
        tmp_path
        / "predictions.jsonl"
    )

    prediction_path.write_text(
        "{}",
        encoding="utf-8",
    )

    config = {
        "inference": {
            "output_path": str(
                prediction_path
            ),
            "overwrite": False,
        },
        "evaluation": {
            "output_dir": str(
                tmp_path
                / "evaluation"
            ),
            "overwrite": False,
        },
        "artifacts": {
            "save_config_snapshot": False,
            "save_run_metadata": False,
        },
    }

    with pytest.raises(
        FileExistsError,
        match="Prediction artifact already exists",
    ):
        validate_output_paths(
            config
        )


def test_validate_output_paths_rejects_non_empty_evaluation_directory(
    tmp_path: Path,
) -> None:
    """Evaluation directory doluysa overwrite kapalıyken reddedildiğini test eder."""

    evaluation_dir = (
        tmp_path
        / "evaluation"
    )

    evaluation_dir.mkdir()

    (
        evaluation_dir
        / "summary.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    config = {
        "inference": {
            "output_path": str(
                tmp_path
                / "predictions.jsonl"
            ),
            "overwrite": False,
        },
        "evaluation": {
            "output_dir": str(
                evaluation_dir
            ),
            "overwrite": False,
        },
        "artifacts": {
            "save_config_snapshot": False,
            "save_run_metadata": False,
        },
    }

    with pytest.raises(
        FileExistsError,
        match="Evaluation output directory is not empty",
    ):
        validate_output_paths(
            config
        )


def test_validate_output_paths_rejects_existing_config_snapshot(
    tmp_path: Path,
) -> None:
    """Existing config snapshot'ın overwrite kapalıyken reddedildiğini test eder."""

    output_dir = (
        tmp_path
        / "outputs"
    )

    output_dir.mkdir()

    (
        output_dir
        / "experiment_config.yaml"
    ).write_text(
        "experiment: baseline\n",
        encoding="utf-8",
    )

    config = {
        "inference": {
            "output_path": str(
                output_dir
                / "predictions.jsonl"
            ),
            "overwrite": False,
        },
        "evaluation": {
            "output_dir": str(
                output_dir
                / "evaluation"
            ),
            "overwrite": False,
        },
        "artifacts": {
            "save_config_snapshot": True,
            "save_run_metadata": False,
        },
    }

    with pytest.raises(
        FileExistsError,
        match="Config snapshot already exists",
    ):
        validate_output_paths(
            config
        )


def test_validate_output_paths_rejects_existing_run_metadata(
    tmp_path: Path,
) -> None:
    """Existing run metadata'nın overwrite kapalıyken reddedildiğini test eder."""

    output_dir = (
        tmp_path
        / "outputs"
    )

    output_dir.mkdir()

    (
        output_dir
        / "run_metadata.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    config = {
        "inference": {
            "output_path": str(
                output_dir
                / "predictions.jsonl"
            ),
            "overwrite": False,
        },
        "evaluation": {
            "output_dir": str(
                output_dir
                / "evaluation"
            ),
            "overwrite": False,
        },
        "artifacts": {
            "save_config_snapshot": False,
            "save_run_metadata": True,
        },
    }

    with pytest.raises(
        FileExistsError,
        match="Run metadata already exists",
    ):
        validate_output_paths(
            config
        )


def test_validate_benchmark_pairs() -> None:
    """Complete EN-AZ pair yapısının doğrulandığını test eder."""

    records = [
        make_record(
            "pair_1_en",
            "pair_1",
            "en",
        ),
        make_record(
            "pair_1_az",
            "pair_1",
            "az",
        ),
        make_record(
            "pair_2_en",
            "pair_2",
            "en",
        ),
        make_record(
            "pair_2_az",
            "pair_2",
            "az",
        ),
    ]

    pair_count = validate_benchmark_pairs(
        records=records,
        source_language="en",
        target_language="az",
        require_complete_pairs=True,
    )

    assert pair_count == 2


def test_validate_benchmark_pairs_rejects_empty_input() -> None:
    """Boş benchmark listesinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Benchmark input contains no records",
    ):
        validate_benchmark_pairs(
            records=[],
            source_language="en",
            target_language="az",
            require_complete_pairs=True,
        )


def test_validate_benchmark_pairs_rejects_task_mismatch() -> None:
    """Aynı pair içindeki task değerleri farklıysa reddedildiğini test eder."""

    records = [
        make_record(
            "pair_1_en",
            "pair_1",
            "en",
            task="reasoning",
        ),
        make_record(
            "pair_1_az",
            "pair_1",
            "az",
            task="factual_knowledge",
        ),
    ]

    with pytest.raises(
        ValueError,
    ):
        validate_benchmark_pairs(
            records=records,
            source_language="en",
            target_language="az",
            require_complete_pairs=True,
        )


def test_validate_benchmark_pairs_rejects_incomplete_pair() -> None:
    """Complete pair zorunluyken eksik language kaydının reddedildiğini test eder."""

    records = [
        make_record(
            "pair_1_en",
            "pair_1",
            "en",
        )
    ]

    with pytest.raises(
        ValueError,
    ):
        validate_benchmark_pairs(
            records=records,
            source_language="en",
            target_language="az",
            require_complete_pairs=True,
        )


def test_validate_benchmark_pairs_allows_incomplete_pair_when_disabled() -> None:
    """Complete pair kontrolü kapalıysa tek-language pair'in kabul edildiğini test eder."""

    records = [
        make_record(
            "pair_1_en",
            "pair_1",
            "en",
        )
    ]

    pair_count = validate_benchmark_pairs(
        records=records,
        source_language="en",
        target_language="az",
        require_complete_pairs=False,
    )

    assert pair_count == 1


@patch(
    "src.experiments.preflight.validate_output_paths"
)
@patch(
    "src.experiments.preflight.load_records"
)
@patch(
    "src.experiments.preflight.load_model_config"
)
def test_run_preflight(
    mock_load_model_config: MagicMock,
    mock_load_records: MagicMock,
    mock_validate_output_paths: MagicMock,
    tmp_path: Path,
) -> None:
    """Tüm baseline preflight zincirinin başarılı çalıştığını test eder."""

    config = make_config(
        tmp_path
    )

    model_config_path = Path(
        config[
            "model"
        ][
            "config_path"
        ]
    )

    benchmark_path = Path(
        config[
            "benchmark"
        ][
            "input_path"
        ]
    )

    model_config_path.write_text(
        "model: {}\n",
        encoding="utf-8",
    )

    benchmark_path.write_text(
        "",
        encoding="utf-8",
    )

    model_config = MagicMock()
    model_config.model_name = "test-model"

    mock_load_model_config.return_value = (
        model_config
    )

    mock_load_records.return_value = [
        make_record(
            "pair_1_en",
            "pair_1",
            "en",
        ),
        make_record(
            "pair_1_az",
            "pair_1",
            "az",
        ),
    ]

    result = run_preflight(
        config
    )

    assert isinstance(
        result,
        PreflightResult,
    )

    assert result.experiment_name == "baseline"
    assert result.benchmark_record_count == 2
    assert result.pair_count == 1
    assert result.source_language == "en"
    assert result.target_language == "az"
    assert result.model_name == "test-model"
    assert result.ready is True

    mock_validate_output_paths.assert_called_once_with(
        config
    )


def test_run_preflight_rejects_missing_model_config(
    tmp_path: Path,
) -> None:
    """Model config dosyası yoksa preflight'ın durduğunu test eder."""

    config = make_config(
        tmp_path
    )

    benchmark_path = Path(
        config[
            "benchmark"
        ][
            "input_path"
        ]
    )

    benchmark_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Model config not found",
    ):
        run_preflight(
            config
        )


def test_run_preflight_rejects_missing_benchmark(
    tmp_path: Path,
) -> None:
    """Benchmark dosyası yoksa preflight'ın durduğunu test eder."""

    config = make_config(
        tmp_path
    )

    model_config_path = Path(
        config[
            "model"
        ][
            "config_path"
        ]
    )

    model_config_path.write_text(
        "model: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Benchmark input not found",
    ):
        run_preflight(
            config
        )


@patch(
    "src.experiments.preflight.load_model_config"
)
def test_run_preflight_rejects_empty_model_name(
    mock_load_model_config: MagicMock,
    tmp_path: Path,
) -> None:
    """Boş model identifier'ın preflight tarafından reddedildiğini test eder."""

    config = make_config(
        tmp_path
    )

    model_config_path = Path(
        config[
            "model"
        ][
            "config_path"
        ]
    )

    benchmark_path = Path(
        config[
            "benchmark"
        ][
            "input_path"
        ]
    )

    model_config_path.write_text(
        "model: {}\n",
        encoding="utf-8",
    )

    benchmark_path.write_text(
        "",
        encoding="utf-8",
    )

    model_config = MagicMock()
    model_config.model_name = "   "

    mock_load_model_config.return_value = (
        model_config
    )

    with pytest.raises(
        ValueError,
        match="Model name cannot be empty",
    ):
        run_preflight(
            config
        )
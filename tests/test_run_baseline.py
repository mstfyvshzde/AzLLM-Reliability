"""Baseline experiment pipeline yardımcılarını test eder."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

from src.experiments.config_validation import (
    validate_experiment_config,
)
from src.experiments.run_baseline import (
    build_run_metadata,
    load_experiment_config,
    run_baseline_experiment,
    save_config_snapshot,
    save_run_metadata,
    set_reproducibility_seed,
)


def make_experiment_config(
    tmp_path: Path,
) -> dict:
    """Testler için minimal geçerli baseline experiment config oluşturur."""

    return {
        "experiment": {
            "name": "baseline",
            "description": "Test baseline experiment.",
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
            "capability": {
                "enabled": True,
            },
            "reliability": {
                "enabled": True,
            },
            "overwrite": False,
        },
        "artifacts": {
            "save_config_snapshot": True,
            "save_run_metadata": True,
        },
    }


def test_load_experiment_config(
    tmp_path: Path,
) -> None:
    """Geçerli YAML experiment config'in yüklendiğini test eder."""

    config_path = (
        tmp_path
        / "baseline.yaml"
    )

    config = make_experiment_config(
        tmp_path
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
        )

    loaded = load_experiment_config(
        config_path
    )

    assert loaded[
        "experiment"
    ][
        "name"
    ] == "baseline"

    assert loaded[
        "languages"
    ][
        "source"
    ] == "en"


def test_load_experiment_config_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Mevcut olmayan experiment config dosyasını reddeder."""

    with pytest.raises(
        FileNotFoundError,
        match="Experiment config not found",
    ):
        load_experiment_config(
            tmp_path
            / "missing.yaml"
        )


def test_load_experiment_config_rejects_non_mapping(
    tmp_path: Path,
) -> None:
    """YAML root yapısı mapping değilse reddeder."""

    config_path = (
        tmp_path
        / "baseline.yaml"
    )

    config_path.write_text(
        "- baseline\n- experiment\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Experiment config must be a mapping",
    ):
        load_experiment_config(
            config_path
        )


def test_validate_experiment_config() -> None:
    """Geçerli experiment config'in validation'dan geçtiğini test eder."""

    config = {
        "experiment": {
            "name": "baseline",
            "seed": 17,
        },
        "languages": {
            "source": "en",
            "target": "az",
        },
        "model": {
            "config_path": "configs/models/base.yaml",
        },
        "benchmark": {
            "input_path": "data/test.jsonl",
        },
        "inference": {
            "output_path": "outputs/predictions.jsonl",
        },
        "evaluation": {
            "output_dir": "outputs/evaluation",
        },
        "artifacts": {},
    }

    validate_experiment_config(
        config
    )


def test_validate_rejects_missing_section() -> None:
    """Required experiment config section eksikse reddeder."""

    config = {
        "experiment": {
            "name": "baseline",
            "seed": 17,
        }
    }

    with pytest.raises(
        ValueError,
        match="Missing experiment config sections",
    ):
        validate_experiment_config(
            config
        )


def test_validate_rejects_empty_experiment_name(
    tmp_path: Path,
) -> None:
    """Boş experiment name değerini reddeder."""

    config = make_experiment_config(
        tmp_path
    )

    config[
        "experiment"
    ][
        "name"
    ] = ""

    with pytest.raises(
        ValueError,
        match="Experiment name cannot be empty",
    ):
        validate_experiment_config(
            config
        )


def test_validate_rejects_non_integer_seed(
    tmp_path: Path,
) -> None:
    """Integer olmayan seed değerini reddeder."""

    config = make_experiment_config(
        tmp_path
    )

    config[
        "experiment"
    ][
        "seed"
    ] = "17"

    with pytest.raises(
        TypeError,
        match="Experiment seed must be an integer",
    ):
        validate_experiment_config(
            config
        )


def test_validate_rejects_same_languages(
    tmp_path: Path,
) -> None:
    """Source ve target language aynıysa reddeder."""

    config = make_experiment_config(
        tmp_path
    )

    config[
        "languages"
    ][
        "target"
    ] = "en"

    with pytest.raises(
        ValueError,
        match="Source and target languages must be different",
    ):
        validate_experiment_config(
            config
        )


def test_set_reproducibility_seed() -> None:
    """Aynı seed ile aynı PyTorch random değerlerinin üretildiğini test eder."""

    set_reproducibility_seed(
        17
    )

    first = torch.rand(
        3
    )

    set_reproducibility_seed(
        17
    )

    second = torch.rand(
        3
    )

    assert torch.equal(
        first,
        second,
    )


def test_save_config_snapshot(
    tmp_path: Path,
) -> None:
    """Experiment config snapshot'ın YAML olarak kaydedildiğini test eder."""

    output_path = (
        tmp_path
        / "experiment_config.yaml"
    )

    config = {
        "experiment": {
            "name": "baseline",
            "seed": 17,
        }
    }

    save_config_snapshot(
        config=config,
        output_path=output_path,
    )

    loaded = yaml.safe_load(
        output_path.read_text(
            encoding="utf-8",
        )
    )

    assert loaded == config


def test_save_config_snapshot_rejects_existing_file(
    tmp_path: Path,
) -> None:
    """Overwrite kapalıyken existing config snapshot'ı reddeder."""

    output_path = (
        tmp_path
        / "experiment_config.yaml"
    )

    output_path.write_text(
        "existing: true\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="Config snapshot already exists",
    ):
        save_config_snapshot(
            config={
                "experiment": {
                    "name": "baseline",
                }
            },
            output_path=output_path,
        )


def test_build_run_metadata(
    tmp_path: Path,
) -> None:
    """Run metadata'nın gerekli reproducibility alanlarını içerdiğini test eder."""

    experiment_config = make_experiment_config(
        tmp_path
    )

    model_config = {
        "model_name": "test-model",
        "device": "cpu",
    }

    metadata = build_run_metadata(
        experiment_config=experiment_config,
        model_config=model_config,
        prediction_count=10,
    )

    assert metadata[
        "experiment_name"
    ] == "baseline"

    assert metadata[
        "seed"
    ] == 17

    assert metadata[
        "source_language"
    ] == "en"

    assert metadata[
        "target_language"
    ] == "az"

    assert metadata[
        "benchmark_split"
    ] == "test"

    assert metadata[
        "prediction_count"
    ] == 10

    assert metadata[
        "model_config"
    ] == model_config

    assert "timestamp_utc" in metadata

    assert metadata[
        "primary_capability_metric"
    ] == "task_aware"

    assert metadata[
        "primary_paired_capability_metric"
    ] == "paired_task_aware"

    assert metadata[
        "diagnostic_capability_metrics"
    ] == [
        "exact_match",
        "short_answer",
        "semantic_answer",
    ]


def test_save_run_metadata(
    tmp_path: Path,
) -> None:
    """Run metadata JSON artifact'ının kaydedildiğini test eder."""

    output_path = (
        tmp_path
        / "run_metadata.json"
    )

    metadata = {
        "experiment_name": "baseline",
        "seed": 17,
    }

    save_run_metadata(
        metadata=metadata,
        output_path=output_path,
    )

    assert output_path.exists()


def test_save_run_metadata_rejects_existing_file(
    tmp_path: Path,
) -> None:
    """Overwrite kapalıyken existing metadata artifact'ını reddeder."""

    output_path = (
        tmp_path
        / "run_metadata.json"
    )

    output_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="Run metadata already exists",
    ):
        save_run_metadata(
            metadata={
                "seed": 17,
            },
            output_path=output_path,
        )


@patch(
    "src.experiments.run_baseline.run_preflight"
)
@patch(
    "src.experiments.run_baseline.save_run_metadata"
)
@patch(
    "src.experiments.run_baseline.save_config_snapshot"
)
@patch(
    "src.experiments.run_baseline.save_evaluation_artifacts"
)
@patch(
    "src.experiments.run_baseline.evaluate_predictions"
)
@patch(
    "src.experiments.run_baseline.save_predictions"
)
@patch(
    "src.experiments.run_baseline.run_inference"
)
@patch(
    "src.experiments.run_baseline.load_model_and_tokenizer"
)
@patch(
    "src.experiments.run_baseline.load_model_config_dict"
)
@patch(
    "src.experiments.run_baseline.load_model_config"
)
@patch(
    "src.experiments.run_baseline.load_records"
)
def test_run_baseline_experiment(
    mock_load_records: MagicMock,
    mock_load_model_config: MagicMock,
    mock_load_model_config_dict: MagicMock,
    mock_load_model_and_tokenizer: MagicMock,
    mock_run_inference: MagicMock,
    mock_save_predictions: MagicMock,
    mock_evaluate_predictions: MagicMock,
    mock_save_evaluation_artifacts: MagicMock,
    mock_save_config_snapshot: MagicMock,
    mock_save_run_metadata: MagicMock,
    mock_run_preflight: MagicMock,
    tmp_path: Path,
) -> None:
    """Baseline pipeline'ın tüm ana adımlarını doğru çağırdığını test eder."""

    config = make_experiment_config(
        tmp_path
    )

    mock_run_preflight.return_value = MagicMock(
        ready=True
    )

    benchmark_records = [
        MagicMock(),
        MagicMock(),
    ]

    model_config = MagicMock()

    raw_model_config = {
        "model_name": "test-model",
    }

    model = MagicMock()

    tokenizer = MagicMock()

    predictions = [
        MagicMock(),
        MagicMock(),
    ]

    evaluation = {
        "capability_summary": {},
        "reliability_summary": {},
    }

    mock_load_records.return_value = (
        benchmark_records
    )

    mock_load_model_config.return_value = (
        model_config
    )

    mock_load_model_config_dict.return_value = (
        raw_model_config
    )

    mock_load_model_and_tokenizer.return_value = (
        model,
        tokenizer,
    )

    mock_run_inference.return_value = (
        predictions
    )

    mock_evaluate_predictions.return_value = (
        evaluation
    )

    result = run_baseline_experiment(
        config
    )

    mock_run_preflight.assert_called_once_with(
        config
    )

    mock_load_records.assert_called_once()

    mock_load_model_config.assert_called_once()

    mock_load_model_config_dict.assert_called_once()

    mock_load_model_and_tokenizer.assert_called_once_with(
        model_config
    )

    mock_run_inference.assert_called_once_with(
        records=benchmark_records,
        model=model,
        tokenizer=tokenizer,
        config=model_config,
    )

    mock_save_predictions.assert_called_once()

    mock_evaluate_predictions.assert_called_once_with(
        predictions=predictions,
        source_language="en",
        target_language="az",
        semantic_adjudication_decisions=None,
        binary_adjudication_decisions=None,
    )

    mock_save_evaluation_artifacts.assert_called_once()

    mock_save_config_snapshot.assert_called_once()

    mock_save_run_metadata.assert_called_once()

    assert result[
        "predictions"
    ] == predictions

    assert result[
        "evaluation"
    ] == evaluation


@patch(
    "src.experiments.run_baseline.run_preflight"
)
@patch(
    "src.experiments.run_baseline.load_records"
)
def test_run_baseline_rejects_empty_benchmark(
    mock_load_records: MagicMock,
    mock_run_preflight: MagicMock,
    tmp_path: Path,
) -> None:
    """Preflight sonrası boş benchmark input'un reddedildiğini test eder."""

    config = make_experiment_config(
        tmp_path
    )

    mock_run_preflight.return_value = MagicMock(
        ready=True
    )

    mock_load_records.return_value = []

    with pytest.raises(
        ValueError,
        match="Benchmark input contains no records",
    ):
        run_baseline_experiment(
            config
        )

    mock_run_preflight.assert_called_once_with(
        config
    )

def test_run_baseline_skip_evaluation(
    tmp_path: Path,
) -> None:
    """skip_evaluation=True iken evaluation adımları çalışmamalıdır."""
    from unittest.mock import patch, MagicMock

    from src.experiments.run_baseline import (
        run_baseline_experiment,
    )

    config = make_experiment_config(
        tmp_path
    )

    benchmark_records = [
        MagicMock(),
        MagicMock(),
    ]

    predictions = [
        MagicMock(),
        MagicMock(),
    ]

    model_config = MagicMock()
    raw_model_config = {
        "model_name": "test-model",
    }

    model = MagicMock()
    tokenizer = MagicMock()

    with (
        patch(
            "src.experiments.run_baseline.run_preflight"
        ) as mock_run_preflight,
        patch(
            "src.experiments.run_baseline.load_records",
            return_value=benchmark_records,
        ),
        patch(
            "src.experiments.run_baseline.load_model_config",
            return_value=model_config,
        ),
        patch(
            "src.experiments.run_baseline.load_model_config_dict",
            return_value=raw_model_config,
        ),
        patch(
            "src.experiments.run_baseline.load_model_and_tokenizer",
            return_value=(model, tokenizer),
        ),
        patch(
            "src.experiments.run_baseline.run_inference",
            return_value=predictions,
        ),
        patch(
            "src.experiments.run_baseline.save_predictions"
        ),
        patch(
            "src.experiments.run_baseline.evaluate_predictions"
        ) as mock_evaluate_predictions,
        patch(
            "src.experiments.run_baseline.save_evaluation_artifacts"
        ) as mock_save_evaluation_artifacts,
        patch(
            "src.experiments.run_baseline.save_config_snapshot"
        ),
        patch(
            "src.experiments.run_baseline.save_run_metadata"
        ),
    ):
        mock_run_preflight.return_value = MagicMock(
            ready=True
        )

        result = run_baseline_experiment(
            config,
            skip_evaluation=True,
        )

    mock_evaluate_predictions.assert_not_called()
    mock_save_evaluation_artifacts.assert_not_called()

    assert result["predictions"] == predictions
    assert result["evaluation"] is None

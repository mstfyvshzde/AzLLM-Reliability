"""Model YAML config loading davranışını test eder."""

from pathlib import Path

import pytest

from src.models.load_config import (
    load_model_config,
    load_model_config_dict,
)
from src.models.model_config import ModelConfig


def write_config(
    path: Path,
    content: str,
) -> None:
    """Test sırasında geçici YAML config dosyası oluşturur."""

    path.write_text(
        content,
        encoding="utf-8",
    )


def test_load_model_config(tmp_path: Path) -> None:
    """Geçerli YAML dosyasından ModelConfig üretildiğini test eder."""

    config_path = tmp_path / "model.yaml"

    write_config(
        config_path,
        """
model:
  model_name: meta-llama/Llama-3.2-3B-Instruct
  tokenizer_name: null
  revision: null
  device: cpu
  dtype: float32
  trust_remote_code: false
  max_new_tokens: 128
  temperature: 0.0
  do_sample: false
""",
    )

    config = load_model_config(
        config_path
    )

    assert isinstance(
        config,
        ModelConfig,
    )

    assert (
        config.model_name
        == "meta-llama/Llama-3.2-3B-Instruct"
    )

    assert config.tokenizer_name is None
    assert config.device == "cpu"
    assert config.dtype == "float32"
    assert config.max_new_tokens == 128
    assert config.temperature == 0.0
    assert config.do_sample is False


def test_load_model_config_dict(
    tmp_path: Path,
) -> None:
    """Model bölümünün ham Python sözlüğü olarak döndürüldüğünü test eder."""

    config_path = tmp_path / "model.yaml"

    write_config(
        config_path,
        """
model:
  model_name: meta-llama/Llama-3.2-3B-Instruct
  device: cpu
  dtype: float32
""",
    )

    model_section = load_model_config_dict(
        config_path
    )

    assert model_section == {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "device": "cpu",
        "dtype": "float32",
    }


def test_missing_model_config_file_is_rejected(
    tmp_path: Path,
) -> None:
    """Bulunmayan model config dosyasının reddedildiğini test eder."""

    config_path = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Model config not found",
    ):
        load_model_config(
            config_path
        )


def test_non_mapping_yaml_is_rejected(
    tmp_path: Path,
) -> None:
    """YAML root değeri mapping değilse reddedildiğini test eder."""

    config_path = tmp_path / "model.yaml"

    write_config(
        config_path,
        """
- model
- config
""",
    )

    with pytest.raises(
        ValueError,
        match="must contain a YAML mapping",
    ):
        load_model_config(
            config_path
        )


def test_missing_model_section_is_rejected(
    tmp_path: Path,
) -> None:
    """Model bölümü bulunmayan YAML dosyasının reddedildiğini test eder."""

    config_path = tmp_path / "model.yaml"

    write_config(
        config_path,
        """
experiment:
  seed: 17
""",
    )

    with pytest.raises(
        ValueError,
        match="must contain a 'model' mapping",
    ):
        load_model_config(
            config_path
        )


def test_model_section_must_be_mapping(
    tmp_path: Path,
) -> None:
    """Model bölümü mapping değilse reddedildiğini test eder."""

    config_path = tmp_path / "model.yaml"

    write_config(
        config_path,
        """
model: meta-llama/Llama-3.2-3B-Instruct
""",
    )

    with pytest.raises(
        ValueError,
        match="must contain a 'model' mapping",
    ):
        load_model_config(
            config_path
        )


def test_invalid_model_config_is_rejected(
    tmp_path: Path,
) -> None:
    """ModelConfig kurallarını ihlal eden YAML değerlerinin reddedildiğini test eder."""

    config_path = tmp_path / "model.yaml"

    write_config(
        config_path,
        """
model:
  model_name: ""
  device: cpu
  dtype: float32
""",
    )

    with pytest.raises(
        ValueError,
        match="model_name cannot be empty",
    ):
        load_model_config(
            config_path
        )


def test_model_config_defaults_are_applied(
    tmp_path: Path,
) -> None:
    """YAML'da bulunmayan optional alanlarda ModelConfig defaultlarının kullanıldığını test eder."""

    config_path = tmp_path / "model.yaml"

    write_config(
        config_path,
        """
model:
  model_name: meta-llama/Llama-3.2-3B-Instruct
""",
    )

    config = load_model_config(
        config_path
    )

    assert config.device == "auto"
    assert config.dtype == "bfloat16"
    assert config.max_new_tokens == 128
    assert config.temperature == 0.0
    assert config.do_sample is False
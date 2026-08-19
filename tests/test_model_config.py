"""ModelConfig doğrulama ve serialization davranışını test eder."""

import pytest

from src.models.model_config import ModelConfig


def test_minimal_model_config() -> None:
    """Yalnızca model_name ile varsayılan config oluşturulabildiğini test eder."""

    config = ModelConfig(
        model_name="example/model",
    )

    assert config.model_name == "example/model"
    assert config.resolved_tokenizer_name == "example/model"
    assert config.device == "auto"
    assert config.dtype == "bfloat16"
    assert config.max_new_tokens == 128
    assert config.temperature == 0.0
    assert config.do_sample is False


def test_custom_tokenizer_name() -> None:
    """Ayrı tokenizer_name verildiğinde onun kullanıldığını test eder."""

    config = ModelConfig(
        model_name="example/model",
        tokenizer_name="example/tokenizer",
    )

    assert config.resolved_tokenizer_name == "example/tokenizer"


def test_empty_model_name_is_rejected() -> None:
    """Boş model_name değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="model_name cannot be empty",
    ):
        ModelConfig(
            model_name="",
        )


def test_empty_tokenizer_name_is_rejected() -> None:
    """Boş tokenizer_name değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="tokenizer_name cannot be empty",
    ):
        ModelConfig(
            model_name="example/model",
            tokenizer_name="",
        )


def test_invalid_device_is_rejected() -> None:
    """Desteklenmeyen device değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Unsupported device",
    ):
        ModelConfig(
            model_name="example/model",
            device="invalid-device",
        )


def test_invalid_dtype_is_rejected() -> None:
    """Desteklenmeyen dtype değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Unsupported dtype",
    ):
        ModelConfig(
            model_name="example/model",
            dtype="invalid-dtype",
        )


def test_non_positive_max_new_tokens_is_rejected() -> None:
    """Pozitif olmayan max_new_tokens değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="max_new_tokens must be greater than zero",
    ):
        ModelConfig(
            model_name="example/model",
            max_new_tokens=0,
        )


def test_negative_temperature_is_rejected() -> None:
    """Negatif temperature değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="temperature cannot be negative",
    ):
        ModelConfig(
            model_name="example/model",
            temperature=-0.1,
            do_sample=True,
        )


def test_temperature_requires_sampling() -> None:
    """Sampling kapalıyken non-zero temperature kullanımının reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="temperature must be 0.0 when do_sample is False",
    ):
        ModelConfig(
            model_name="example/model",
            temperature=0.7,
            do_sample=False,
        )


def test_sampling_temperature_is_allowed() -> None:
    """Sampling açıkken pozitif temperature değerinin kabul edildiğini test eder."""

    config = ModelConfig(
        model_name="example/model",
        temperature=0.7,
        do_sample=True,
    )

    assert config.temperature == 0.7
    assert config.do_sample is True


def test_to_dict() -> None:
    """ModelConfig nesnesinin doğru sözlüğe dönüştürüldüğünü test eder."""

    config = ModelConfig(
        model_name="example/model",
        tokenizer_name="example/tokenizer",
        revision="main",
        device="cpu",
        dtype="float32",
        trust_remote_code=False,
        max_new_tokens=64,
        temperature=0.0,
        do_sample=False,
    )

    result = config.to_dict()

    assert result == {
        "model_name": "example/model",
        "tokenizer_name": "example/tokenizer",
        "revision": "main",
        "device": "cpu",
        "dtype": "float32",
        "trust_remote_code": False,
        "max_new_tokens": 64,
        "temperature": 0.0,
        "do_sample": False,
    }


def test_from_dict() -> None:
    """Python sözlüğünden ModelConfig oluşturulabildiğini test eder."""

    config = ModelConfig.from_dict(
        {
            "model_name": "example/model",
            "device": "cpu",
            "dtype": "float32",
            "max_new_tokens": 32,
        }
    )

    assert config.model_name == "example/model"
    assert config.device == "cpu"
    assert config.dtype == "float32"
    assert config.max_new_tokens == 32


def test_from_dict_uses_defaults() -> None:
    """Eksik optional alanlarda varsayılan değerlerin kullanıldığını test eder."""

    config = ModelConfig.from_dict(
        {
            "model_name": "example/model",
        }
    )

    assert config.tokenizer_name is None
    assert config.device == "auto"
    assert config.dtype == "bfloat16"
    assert config.max_new_tokens == 128
    assert config.temperature == 0.0
    assert config.do_sample is False


def test_from_dict_rejects_non_mapping() -> None:
    """Mapping olmayan config girdisinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Model config must be a mapping",
    ):
        ModelConfig.from_dict(
            ["example/model"]  # type: ignore[arg-type]
        )


def test_from_dict_requires_string_model_name() -> None:
    """String model_name içermeyen config girdisinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="must contain a string 'model_name'",
    ):
        ModelConfig.from_dict(
            {
                "model_name": 123,
            }
        )
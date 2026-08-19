"""Model ve tokenizer loading yardımcılarını test eder."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.load_model import (
    build_model_kwargs,
    build_tokenizer_kwargs,
    load_causal_model,
    load_model_and_tokenizer,
    load_tokenizer,
    move_model_to_device,
    prepare_tokenizer,
    resolve_torch_dtype,
)
from src.models.model_config import ModelConfig


def make_config(
    **overrides,
) -> ModelConfig:
    """Model loading testleri için örnek ModelConfig oluşturur."""

    values = {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "tokenizer_name": None,
        "revision": None,
        "device": "cpu",
        "dtype": "float32",
        "trust_remote_code": False,
        "max_new_tokens": 128,
        "temperature": 0.0,
        "do_sample": False,
    }

    values.update(overrides)

    return ModelConfig(**values)


def test_resolve_torch_dtype() -> None:
    """Desteklenen dtype string değerlerinin PyTorch dtype'a çözüldüğünü test eder."""

    assert resolve_torch_dtype("float32") is torch.float32
    assert resolve_torch_dtype("float16") is torch.float16
    assert resolve_torch_dtype("bfloat16") is torch.bfloat16


def test_invalid_torch_dtype_is_rejected() -> None:
    """Desteklenmeyen dtype değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Unsupported torch dtype",
    ):
        resolve_torch_dtype("invalid")


def test_build_tokenizer_kwargs() -> None:
    """Tokenizer loading kwargs değerlerinin config'ten üretildiğini test eder."""

    config = make_config(
        revision="main",
        trust_remote_code=True,
    )

    kwargs = build_tokenizer_kwargs(config)

    assert kwargs == {
        "trust_remote_code": True,
        "revision": "main",
    }


def test_build_tokenizer_kwargs_without_revision() -> None:
    """Revision verilmediğinde tokenizer kwargs içine eklenmediğini test eder."""

    config = make_config()

    kwargs = build_tokenizer_kwargs(config)

    assert kwargs == {
        "trust_remote_code": False,
    }


def test_build_model_kwargs_for_auto_device() -> None:
    """Auto device seçiminde device_map kullanıldığını test eder."""

    config = make_config(
        device="auto",
        dtype="bfloat16",
    )

    kwargs = build_model_kwargs(config)

    assert kwargs["dtype"] is torch.bfloat16
    assert kwargs["device_map"] == "auto"
    assert kwargs["trust_remote_code"] is False


def test_build_model_kwargs_for_explicit_device() -> None:
    """Explicit device seçiminde device_map eklenmediğini test eder."""

    config = make_config(
        device="cpu",
        dtype="float32",
    )

    kwargs = build_model_kwargs(config)

    assert kwargs["dtype"] is torch.float32
    assert "device_map" not in kwargs


def test_prepare_tokenizer_sets_pad_token_from_eos() -> None:
    """Pad token yoksa EOS token'ın padding için kullanıldığını test eder."""

    tokenizer = MagicMock()

    tokenizer.pad_token_id = None
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = None

    result = prepare_tokenizer(tokenizer)

    assert result is tokenizer
    assert tokenizer.pad_token == "</s>"


def test_prepare_tokenizer_keeps_existing_pad_token() -> None:
    """Mevcut pad token değerinin değiştirilmediğini test eder."""

    tokenizer = MagicMock()

    tokenizer.pad_token_id = 0
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"

    result = prepare_tokenizer(tokenizer)

    assert result is tokenizer
    assert tokenizer.pad_token == "<pad>"


def test_prepare_tokenizer_without_eos_token() -> None:
    """EOS token bulunmuyorsa tokenizer'ın değiştirilmeden döndüğünü test eder."""

    tokenizer = MagicMock()

    tokenizer.pad_token_id = None
    tokenizer.eos_token = None
    tokenizer.pad_token = None

    result = prepare_tokenizer(tokenizer)

    assert result is tokenizer
    assert tokenizer.pad_token is None


def test_move_model_to_auto_device_does_not_call_to() -> None:
    """Auto device seçiminde model.to çağrılmadığını test eder."""

    model = MagicMock()

    result = move_model_to_device(
        model,
        "auto",
    )

    assert result is model
    model.to.assert_not_called()


def test_move_model_to_cpu() -> None:
    """Explicit CPU seçiminde model.to çağrıldığını test eder."""

    model = MagicMock()

    result = move_model_to_device(
        model,
        "cpu",
    )

    assert result is model

    model.to.assert_called_once_with(
        torch.device("cpu")
    )


@patch("src.models.load_model.AutoTokenizer.from_pretrained")
def test_load_tokenizer(
    mock_from_pretrained,
) -> None:
    """Tokenizer'ın doğru identifier ve kwargs ile yüklendiğini test eder."""

    tokenizer = MagicMock()

    tokenizer.pad_token_id = None
    tokenizer.eos_token = "</s>"

    mock_from_pretrained.return_value = tokenizer

    config = make_config(
        tokenizer_name="example/custom-tokenizer",
        revision="main",
    )

    result = load_tokenizer(config)

    mock_from_pretrained.assert_called_once_with(
        "example/custom-tokenizer",
        trust_remote_code=False,
        revision="main",
    )

    assert result is tokenizer
    assert tokenizer.pad_token == "</s>"


@patch("src.models.load_model.AutoModelForCausalLM.from_pretrained")
def test_load_causal_model(
    mock_from_pretrained,
) -> None:
    """Causal modelin config ile yüklenip eval mode'a geçirildiğini test eder."""

    model = MagicMock()
    mock_from_pretrained.return_value = model

    config = make_config(
        device="cpu",
        dtype="float32",
    )

    result = load_causal_model(config)

    mock_from_pretrained.assert_called_once_with(
        "meta-llama/Llama-3.2-3B-Instruct",
        dtype=torch.float32,
        trust_remote_code=False,
    )

    model.to.assert_called_once_with(
        torch.device("cpu")
    )

    model.eval.assert_called_once()

    assert result is model


@patch("src.models.load_model.load_causal_model")
@patch("src.models.load_model.load_tokenizer")
def test_load_model_and_tokenizer(
    mock_load_tokenizer,
    mock_load_causal_model,
) -> None:
    """Model ve tokenizer'ın aynı config üzerinden birlikte yüklendiğini test eder."""

    tokenizer = MagicMock()
    model = MagicMock()

    mock_load_tokenizer.return_value = tokenizer
    mock_load_causal_model.return_value = model

    config = make_config()

    loaded_model, loaded_tokenizer = load_model_and_tokenizer(
        config
    )

    mock_load_tokenizer.assert_called_once_with(
        config
    )

    mock_load_causal_model.assert_called_once_with(
        config
    )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
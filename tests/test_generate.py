"""LLM generation yardımcı fonksiyonlarını test eder."""

from unittest.mock import MagicMock

import pytest
import torch

from src.models.generate import (
    DEFAULT_SYSTEM_PROMPT,
    build_generation_kwargs,
    build_messages,
    decode_generated_tokens,
    generate_response,
    get_model_input_device,
    tokenize_messages,
)
from src.models.model_config import ModelConfig


def make_config(
    **overrides,
) -> ModelConfig:
    """Generation testlerinde kullanılacak örnek ModelConfig oluşturur."""

    values = {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "tokenizer_name": None,
        "revision": None,
        "device": "cpu",
        "dtype": "float32",
        "trust_remote_code": False,
        "max_new_tokens": 64,
        "temperature": 0.0,
        "do_sample": False,
    }

    values.update(overrides)

    return ModelConfig(**values)


def test_build_messages_with_system_prompt() -> None:
    """System ve user message yapısının doğru oluşturulduğunu test eder."""

    messages = build_messages(
        prompt="What is 2 + 2?",
    )

    assert messages == [
        {
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
    ]


def test_build_messages_without_system_prompt() -> None:
    """System prompt kapatıldığında yalnızca user message üretildiğini test eder."""

    messages = build_messages(
        prompt="What is 2 + 2?",
        system_prompt=None,
    )

    assert messages == [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        }
    ]


def test_empty_prompt_is_rejected() -> None:
    """Boş prompt değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="Prompt cannot be empty",
    ):
        build_messages(
            prompt="   ",
        )


def test_empty_system_prompt_is_rejected() -> None:
    """Boş system prompt değerinin reddedildiğini test eder."""

    with pytest.raises(
        ValueError,
        match="system_prompt cannot be empty",
    ):
        build_messages(
            prompt="Question",
            system_prompt="   ",
        )


def test_build_generation_kwargs_deterministic() -> None:
    """Deterministic generation ayarlarının doğru oluşturulduğunu test eder."""

    config = make_config()

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    kwargs = build_generation_kwargs(
        config,
        tokenizer,
    )

    assert kwargs == {
        "max_new_tokens": 64,
        "do_sample": False,
        "pad_token_id": 0,
        "eos_token_id": 2,
    }


def test_build_generation_kwargs_with_sampling() -> None:
    """Sampling açıkken temperature değerinin kwargs içine eklendiğini test eder."""

    config = make_config(
        do_sample=True,
        temperature=0.7,
    )

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    kwargs = build_generation_kwargs(
        config,
        tokenizer,
    )

    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.7


def test_generation_kwargs_skip_missing_special_tokens() -> None:
    """Pad veya EOS token yoksa generation kwargs içine eklenmediğini test eder."""

    config = make_config()

    tokenizer = MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = None

    kwargs = build_generation_kwargs(
        config,
        tokenizer,
    )

    assert "pad_token_id" not in kwargs
    assert "eos_token_id" not in kwargs


def test_get_model_input_device() -> None:
    """Model input device değerinin model.device üzerinden alındığını test eder."""

    model = MagicMock()
    model.device = torch.device("cpu")

    device = get_model_input_device(
        model
    )

    assert device == torch.device("cpu")


def test_tokenize_messages() -> None:
    """Chat template çıktısının model device üzerine taşındığını test eder."""

    tokenizer = MagicMock()

    input_ids = MagicMock()
    attention_mask = MagicMock()

    moved_input_ids = MagicMock()
    moved_attention_mask = MagicMock()

    input_ids.to.return_value = moved_input_ids
    attention_mask.to.return_value = moved_attention_mask

    tokenizer.apply_chat_template.return_value = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    messages = [
        {
            "role": "user",
            "content": "Question",
        }
    ]

    device = torch.device("cpu")

    result = tokenize_messages(
        messages=messages,
        tokenizer=tokenizer,
        device=device,
    )

    tokenizer.apply_chat_template.assert_called_once_with(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids.to.assert_called_once_with(
        device
    )

    attention_mask.to.assert_called_once_with(
        device
    )

    assert result == {
        "input_ids": moved_input_ids,
        "attention_mask": moved_attention_mask,
    }


def test_decode_generated_tokens() -> None:
    """Input tokenlarının çıkarılıp yalnızca yeni tokenların decode edildiğini test eder."""

    tokenizer = MagicMock()
    tokenizer.decode.return_value = "  4  "

    generated_ids = torch.tensor(
        [
            [
                10,
                11,
                12,
                20,
                21,
            ]
        ]
    )

    response = decode_generated_tokens(
        generated_ids=generated_ids,
        input_length=3,
        tokenizer=tokenizer,
    )

    tokenizer.decode.assert_called_once()

    decoded_token_ids = tokenizer.decode.call_args.args[0]

    assert torch.equal(
        decoded_token_ids,
        torch.tensor([20, 21]),
    )

    assert tokenizer.decode.call_args.kwargs == {
        "skip_special_tokens": True,
    }

    assert response == "4"


def test_generate_response() -> None:
    """Tek prompt için tokenize-generation-decode akışının çalıştığını test eder."""

    config = make_config()

    model = MagicMock()
    model.device = torch.device("cpu")

    tokenizer = MagicMock()

    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    input_ids = torch.tensor(
        [
            [
                10,
                11,
                12,
            ]
        ]
    )

    attention_mask = torch.tensor(
        [
            [
                1,
                1,
                1,
            ]
        ]
    )

    tokenizer.apply_chat_template.return_value = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    model.generate.return_value = torch.tensor(
        [
            [
                10,
                11,
                12,
                20,
                21,
            ]
        ]
    )

    tokenizer.decode.return_value = "4"

    response = generate_response(
        prompt="What is 2 + 2?",
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    model.generate.assert_called_once()

    generation_call = model.generate.call_args.kwargs

    assert torch.equal(
        generation_call["input_ids"],
        input_ids,
    )

    assert torch.equal(
        generation_call["attention_mask"],
        attention_mask,
    )

    assert generation_call["max_new_tokens"] == 64
    assert generation_call["do_sample"] is False
    assert generation_call["pad_token_id"] == 0
    assert generation_call["eos_token_id"] == 2

    tokenizer.decode.assert_called_once()

    assert response == "4"
"""Causal language model ve tokenizer yükleme işlemlerini yönetir.

Bu modül ModelConfig tarafından tanımlanan ayarları kullanarak Hugging Face
üzerinden veya local bir model klasöründen tokenizer ve causal language model
yükler.

Model yükleme davranışı merkezi tutulur. Böylece baseline evaluation,
reliability analysis ve adaptation sonrası evaluation aşamalarının tamamı
aynı model-loading protokolünü kullanabilir.
"""


from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.models.model_config import ModelConfig


TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_torch_dtype(
    dtype: str,
) -> torch.dtype:
    """String dtype değerini karşılık gelen PyTorch dtype nesnesine dönüştürür.

    Örnek:
        "float32"  -> torch.float32
        "float16"  -> torch.float16
        "bfloat16" -> torch.bfloat16

    Desteklenmeyen dtype değeri için ValueError oluşturur.
    """

    try:
        return TORCH_DTYPES[dtype]

    except KeyError as error:
        raise ValueError(
            f"Unsupported torch dtype '{dtype}'. "
            f"Expected one of: {sorted(TORCH_DTYPES)}"
        ) from error


def build_tokenizer_kwargs(
    config: ModelConfig,
) -> dict[str, Any]:
    """Tokenizer yüklenirken Hugging Face'e gönderilecek ek ayarları oluşturur.

    `kwargs`, keyword arguments demektir. Yani fonksiyona isimleriyle verilen
    ek parametreleri bir sözlükte toplar.

    Örnek config:
    trust_remote_code = False
    revision = "main"

    Oluşan kwargs:
    {
        "trust_remote_code": False,
        "revision": "main"
    }

    Sonra şöyle kullanılır:
    AutoTokenizer.from_pretrained(
        model_name,
        **kwargs
    )

    Buradaki `**kwargs`, sözlükteki değerleri ayrı parametreler gibi gönderir.
    """

    kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
    }

    if config.revision is not None:
        kwargs["revision"] = config.revision

    return kwargs


def build_model_kwargs(
    config: ModelConfig,
) -> dict[str, Any]:
    """Model yüklenirken Hugging Face'e gönderilecek ek ayarları oluşturur.

    `kwargs`, modelin precision, custom code izni, revision ve device placement
    ayarlarını bir sözlükte toplar.

    Örnek config:
    dtype = "bfloat16"
    trust_remote_code = False
    revision = "main"
    device = "auto"

    Oluşan kwargs:
    {
        "dtype": torch.bfloat16,
        "trust_remote_code": False,
        "revision": "main",
        "device_map": "auto"
    }

    Sonra bu ayarlar model yüklenirken kullanılır:
    AutoModelForCausalLM.from_pretrained(
        model_name,
        **kwargs
    )
    """

    kwargs: dict[str, Any] = {
        "dtype": resolve_torch_dtype(config.dtype),
        "trust_remote_code": config.trust_remote_code,
    }

    if config.revision is not None:
        kwargs["revision"] = config.revision

    if config.device == "auto":
        kwargs["device_map"] = "auto"

    return kwargs


def prepare_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedTokenizerBase:
    """Tokenizer'ı generation için hazırlar.

    Bazı causal language model tokenizer'larında `pad_token` tanımlı olmayabilir.
    Bu durumda `eos_token` varsa padding token olarak onu kullanır.

    Örnek:
        pad_token = None
        eos_token = "</s>"

    Sonuç:
        pad_token = "</s>"

    Böylece batch generation sırasında padding işlemi sorun çıkarmadan yapılabilir.

    pad_token → Farklı uzunluktaki metinleri aynı uzunluğa getirmek için eklenen doldurma token'ı.
    Örnek:
    ["Hello", "world", "<pad>", "<pad>"]

    eos_token → Metnin/cevabın bittiğini modele gösteren "end of sequence" token'ı.
    Örnek:
    ["Hello", "world", "</s>"]

    Kısaca:
    <pad> → boşluğu doldurur
    </s>  → metin burada bitti der
    """

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def move_model_to_device(
    model: PreTrainedModel,
    device: str,
) -> PreTrainedModel:
    """Modeli explicit olarak belirtilen execution device üzerine taşır.

    device="auto" durumunda placement zaten from_pretrained sırasında
    device_map="auto" ile yönetildiği için model yeniden taşınmaz.

    cpu, cuda veya mps değerlerinde model ilgili PyTorch device üzerine
    taşınır.
    """

    if device == "auto":
        return model

    model.to(
        torch.device(device)
    )

    return model


def load_tokenizer(
    config: ModelConfig,
) -> PreTrainedTokenizerBase:
    """ModelConfig kullanarak tokenizer yükler ve generation için hazırlar.

    tokenizer_name verilmişse o identifier kullanılır. Aksi durumda
    ModelConfig.resolved_tokenizer_name üzerinden model_name kullanılır.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        config.resolved_tokenizer_name,
        **build_tokenizer_kwargs(config),
    )

    return prepare_tokenizer(
        tokenizer
    )


def load_causal_model(
    config: ModelConfig,
) -> PreTrainedModel:
    """ModelConfig kullanarak causal language model yükler.

    Model Hugging Face AutoModelForCausalLM üzerinden yüklenir.

    device="auto":
        device_map="auto" ile otomatik placement kullanılır.

    device="cpu", "cuda" veya "mps":
        model önce yüklenir, ardından explicit olarak seçilen cihaza taşınır.

    Model evaluation amacıyla kullanılacağı için yükleme sonunda eval mode'a
    geçirilir.
    """

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **build_model_kwargs(config),
    )

    model = move_model_to_device(
        model,
        config.device,
    )

    model.eval()

    return model


def load_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[Any, Any]:
    """Backend'e göre model ve tokenizer yükler.

    transformers:
        Hugging Face / PyTorch loading pipeline kullanılır.

    mlx:
        mlx_lm üzerinden quantized MLX model yüklenir.
        Immutable revision doğrudan MLX loader'a geçirilir.
    """

    if config.backend == "mlx":
        from mlx_lm import load as mlx_load

        model, tokenizer = mlx_load(
            config.model_name,
            revision=config.revision,
        )

        return model, tokenizer

    model = load_causal_model(
        config
    )

    tokenizer = load_tokenizer(
        config
    )

    return model, tokenizer


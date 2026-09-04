"""LLM generation ve tek-prompt inference işlemlerini yönetir.

Bu modül tokenizer ve causal language model kullanarak benchmark sorularından
model cevapları üretir.

Instruction-tuned chat modellerinde tokenizer'ın chat template yapısı
kullanılır. Böylece prompt doğrudan ham metin olarak verilmek yerine modelin
eğitim sırasında beklediği conversation formatına dönüştürülür.

Generation sonunda yalnızca model tarafından üretilen yeni tokenlar decode
edilir; input prompt cevabın içine tekrar dahil edilmez.
"""


from __future__ import annotations

from typing import Any

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase
)

from src.models.model_config import ModelConfig


# DEFAULT_SYSTEM_PROMPT → Modele cevap verirken izlemesi gereken genel talimatı verir.
# Örnek: doğru, kısa ve gereksiz açıklama yapmadan cevapla.
DEFAULT_SYSTEM_PROMPT = (
    "Answer the user's question accurately and concisely. "
    "Do not add unnecessary explanation."
)



def build_messages(
    prompt: str,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT
) -> list[dict[str, str]]:
    """Tek bir benchmark prompt'unu chat message yapısına dönüştürür.

    Instruction-tuned modeller genellikle role/content çiftlerinden oluşan
    conversation formatı bekler.

    Örnek:

        prompt:
            "What is 2 + 2?"

        Sonuç:
            [
                {
                    "role": "system",
                    "content": "Answer the user's question..."
                },
                {
                    "role": "user",
                    "content": "What is 2 + 2?"
                }
            ]

    system_prompt None olduğunda yalnızca user message oluşturulur.

    Boş prompt verilirse ValueError oluşturur.
    """

    if not prompt.strip():
        raise ValueError(
            "Prompt cannot be empty."
        )

    messages: list[dict[str, str]] = []

    if system_prompt is not None:
        if not system_prompt.strip():
            raise ValueError(
                "system_prompt cannot be empty when provided."
            )

        messages.append(
            {
                "role": "system",
                "content": system_prompt
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    return messages



def build_generation_kwargs(
    config: ModelConfig,
    tokenizer: PreTrainedTokenizerBase
) -> dict[str, Any]:
    """Model.generate çağrısında kullanılacak generation ayarlarını oluşturur.

    Deterministic baseline evaluation için varsayılan config:

        do_sample = False
        temperature = 0.0

    Sampling kapalı olduğunda temperature model.generate çağrısına verilmez.
    Böylece Transformers tarafından sampling kullanılmadığı halde temperature
    tanımlandığına dair gereksiz warning üretilmez.

    tokenizer içinde pad ve EOS token kimlikleri mevcutsa generation kwargs
    içine eklenir.
    """

    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample
    }

    # do_sample → Modelin cevabı üretirken rastgele sampling kullanıp kullanmayacağını belirler.
    # False → Daha deterministik, aynı input için benzer/aynı cevap.
    # True  → Olasılıklara göre seçim yapar, cevaplar daha değişken olabilir.
    if config.do_sample:
        kwargs["temperature"] = config.temperature

    if tokenizer.pad_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id

    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id

    return kwargs



def get_model_input_device(
    model: PreTrainedModel
) -> torch.device:
    """Tokenized input'un taşınacağı model device değerini döndürür.

    Normal CPU, CUDA veya MPS modellerinde model.device kullanılır.

    device_map ile dağıtılmış modellerde de model.device çoğu standart
    causal language model için giriş tensorlarının gönderileceği ana device
    değerini sağlar.
    """

    return model.device



def tokenize_messages(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device
) -> dict[str, torch.Tensor]:
    """Chat mesajlarını modelin anlayabileceği tensor inputlarına dönüştürür.

    Örnek messages:
    [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "What is 2 + 2?"}
    ]

    Tokenizer bu mesajları modelin chat template formatına çevirir ve
    `input_ids` ile `attention_mask` üretir.

    Örnek sonuç:
    {
        "input_ids": tensor(...),
        "attention_mask": tensor(...)
    }

    `add_generation_prompt=True`, model cevabının başlayacağı yeri ekler.
    Oluşan tensorlar inference öncesinde belirtilen device'a taşınır.
    """

    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt',
        return_dict=True
    )

    return {
        key: value.to(device)
        for key, value in tokenized.items()
    }


def decode_generated_tokens(
    generated_ids: torch.Tensor,
    input_length: int,
    tokenizer: PreTrainedTokenizerBase
) -> str:
    """Model çıktısından yalnızca yeni üretilen cevabı metne dönüştürür.

    Generation çıktısı hem prompt tokenlarını hem de modelin ürettiği yeni
    tokenları içerir. Bu fonksiyon `input_length` kadar olan prompt kısmını atar
    ve sadece yeni tokenları decode eder.

    Örnek:
    generated_ids = [prompt tokenları + cevap tokenları]
        input_length = 10

    Sonuç:
    ilk 10 token atılır
    yalnızca cevap tokenları decode edilir

    Special tokenlar çıkarılır ve temizlenmiş cevap string olarak döndürülür.

    decode → Token ID'lerini tekrar okunabilir metne çevirir.

    Örnek:
    [15496, 995] → "Hello world"
    """

    new_token_ids = generated_ids[0, input_length:]

    response = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True
    )

    return response.strip()



def generate_response(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: ModelConfig,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT
) -> str:
    """Tek bir prompt için model cevabı üretir.

    Prompt'u chat formatına çevirir, tokenize eder, modeli çalıştırır ve
    yalnızca modelin ürettiği yeni cevabı metin olarak döndürür.

    Örnek:
    prompt:
        "What is 2 + 2?"

    Model generation:
        [prompt tokenları] + [cevap tokenları]

    Sonuç:
        "4"

    Akış:
    1. Prompt chat message formatına çevrilir.
    2. Mesajlar tokenlara dönüştürülür.
    3. Tokenlar model device'ına taşınır.
    4. model.generate() ile cevap üretilir.
    5. Prompt tokenları çıktıdan çıkarılır.
    6. Kalan tokenlar decode edilerek cevap döndürülür.

    `torch.inference_mode()` sayesinde gradient hesaplanmaz ve evaluation
    inference daha verimli çalışır.
    """

    messages = build_messages(
        prompt=prompt,
        system_prompt=system_prompt
    )

    if config.backend == "mlx":
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        mlx_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampler = make_sampler(
            temp=config.temperature,
        )

        response = mlx_generate(
            model,
            tokenizer,
            prompt=mlx_prompt,
            max_tokens=config.max_new_tokens,
            sampler=sampler,
            verbose=False,
        )

        return response.strip()

    device = get_model_input_device(model)

    model_inputs = tokenize_messages(
        messages=messages,
        tokenizer=tokenizer,
        device=device
    )

    input_length = model_inputs[
        'input_ids'
    ].shape[-1]

    generation_kwargs = build_generation_kwargs(
        config=config,
        tokenizer=tokenizer
    )

    # torch.inference_mode() → Gradient hesaplamayı kapatır; inference daha hızlı ve hafif olur.
    # model.generate(...) → Modele inputları ve generation ayarlarını verip cevap tokenlarını üretir.
    # **model_inputs → input_ids, attention_mask gibi model girdilerini açar.
    # **generation_kwargs → max_new_tokens, do_sample gibi generation ayarlarını açar.
    # generated_ids → Modelin ürettiği token ID'lerini tutar.
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            **generation_kwargs
        )

    return decode_generated_tokens(
        generated_ids=generated_ids,
        input_length=input_length,
        tokenizer=tokenizer,
    )
"""Model YAML config dosyalarını yükler ve ModelConfig nesnesine dönüştürür.

Bu modül model ayarlarının Python koduna hard-code edilmesini engeller.
Model adı, tokenizer, device, dtype ve generation ayarları YAML dosyasından
okunur ve ModelConfig üzerinden doğrulanır.
"""


from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.models.model_config import ModelConfig

def load_model_config(
    config_path: Path
) -> ModelConfig:
    """YAML dosyasından doğrulanmış ModelConfig nesnesi oluşturur.

    Beklenen YAML yapısı:
    model_section ->
        model:
            model_name: meta-llama/Llama-3.2-3B-Instruct
            tokenizer_name: null
            revision: null
            device: auto
            dtype: bfloat16
            trust_remote_code: false
            max_new_tokens: 128
            temperature: 0.0
            do_sample: false

    Dosya bulunamazsa FileNotFoundError oluşturur.

    YAML içeriği geçerli bir mapping değilse veya `model` bölümü
    bulunmuyorsa ValueError oluşturur.

    `model` bölümündeki alanların doğrulanması ModelConfig tarafından yapılır.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {config_path}"
        )

    with config_path.open(
        'r',
        encoding='utf-8'
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Model config file must contain a YAML mapping."
        )

    model_section = config.get('model')

    if not isinstance(model_section, dict):
        raise ValueError(
            "Model config must contain a 'model' mapping."
        )

    return ModelConfig.from_dict(
        model_section
    )



def load_model_config_dict(
    config_path: Path
) -> dict[str, Any]:
    """YAML dosyasındaki model bölümünü ham Python sözlüğü olarak döndürür.

    Bu yardımcı fonksiyon özellikle experiment logging ve reproducibility
    çıktılarında model config değerlerinin olduğu gibi saklanması için
    kullanılabilir.

    Model bölümü bulunmazsa veya geçerli bir mapping değilse ValueError
    oluşturur.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {config_path}"
        )

    with config_path.open(
        'r',
        encoding='utf-8'
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Model config file must contain a YAML mapping."
        )

    model_section = config.get('model')

    if not isinstance(model_section, dict):
        raise ValueError(
            "Model config must contain a 'model' mapping."
        )

    return model_section
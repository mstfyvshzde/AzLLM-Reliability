"""Model yükleme ve inference ayarlarını temsil eden config yapısını tanımlar.

Bu modül model inference ve evaluation aşamalarında kullanılacak temel model
ayarlarını merkezi bir veri yapısında toplar.

Amaç, model adı, tokenizer, device, precision ve generation davranışı gibi
ayarların kod içine dağılmasını engellemek ve deneylerin reproducible şekilde
çalıştırılmasını sağlamaktır.
"""


from __future__ import annotations

# dataclass → ModelConfig gibi veri taşıyan sınıflarda __init__ gibi temel metodları otomatik oluşturur.
from dataclasses import dataclass
from typing import Any


# VALID_DTYPES → Modelin hangi sayısal precision türlerinde çalışabileceğini belirtir.
# float32   → Daha yüksek precision, daha fazla bellek kullanır.
# float16   → Daha az bellek, GPU'larda daha hızlı olabilir.
# bfloat16  → float16'ya benzer ama geniş sayı aralığı nedeniyle LLM'lerde sık kullanılır.
VALID_DTYPES = {
    "float32",
    "float16",
    "bfloat16"
}


# VALID_DEVICES → Modelin hangi cihazda çalıştırılabileceğini belirtir.
# auto → Cihazı sistem otomatik seçer.
# cpu  → İşlemci üzerinde çalışır.
# cuda → NVIDIA GPU üzerinde çalışır.
# mps  → Apple Silicon GPU üzerinde çalışır.
VALID_DEVICES = {
    "auto",
    "cpu",
    "cuda",
    "mps"
}



@dataclass(frozen=True)
class ModelConfig:
    """Tek bir LLM evaluation modeli için temel yapılandırmayı temsil eder.

    Alanlar:
        model_name:
            Hugging Face veya local model identifier değeri.

        tokenizer_name:
            Model tokenizer'ı farklı bir kaynaktan yüklenecekse kullanılacak
            tokenizer identifier değeri. None ise model_name kullanılır.

        revision:
            Model repository içindeki branch, tag veya commit revision değeri.

        device:
            Modelin çalıştırılacağı cihaz seçimi.

        dtype:
            Model parametrelerinin kullanılacağı precision türü.

        trust_remote_code:
            Hugging Face model repository içindeki custom Python kodunun
            çalıştırılmasına izin verilip verilmediğini belirler.

        max_new_tokens:
            Generation sırasında üretilebilecek maksimum yeni token sayısı.

        temperature:
            Sampling sırasında kullanılacak temperature değeri.

        do_sample:
            Sampling tabanlı generation kullanılıp kullanılmayacağını belirler.
    """

    model_name: str
    tokenizer_name: str | None = None
    revision: str | None = None
    device: str = 'auto'
    dtype: str = 'bfloat16'
    trust_remote_code: bool = False
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False

    def __post_init__(self) -> None:
        """ModelConfig oluşturulduktan hemen sonra ayarların geçerli olup olmadığını kontrol eder.

    `__post_init__`, dataclass nesnesi oluşturulduktan sonra otomatik çalışır.

    Kontrol edilenler:
    - model_name boş mu?
    - tokenizer_name verilmişse boş mu?
    - device desteklenen cihazlardan biri mi?
    - dtype desteklenen precision türlerinden biri mi?
    - max_new_tokens 0'dan büyük mü?
    - temperature negatif mi?
    - do_sample=False iken temperature yanlışlıkla 0'dan farklı mı?

    Örnek geçerli config:
    ModelConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="mps",
        dtype="float16",
        max_new_tokens=128,
        temperature=0.0,
        do_sample=False,
    )

    Örnek hatalı config:
    ModelConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="gpu"
    )

    `gpu` VALID_DEVICES içinde olmadığı için ValueError oluşturur.
    """

        if not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if self.tokenizer_name is not None:
            if not self.tokenizer_name.strip():
                raise ValueError(
                    "tokenizer_name cannot be empty when provided."
                )

        if self.device not in VALID_DEVICES:
            raise ValueError(
                f"Unsupported device '{self.device}'. "
                f"Expected one of: {sorted(VALID_DEVICES)}"
            )

        if self.dtype not in VALID_DTYPES:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}'. "
                f"Expected one of: {sorted(VALID_DTYPES)}"
            )

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        if self.temperature < 0.0:
            raise ValueError(
                "temperature cannot be negative."
            )

        if not self.do_sample and self.temperature != 0.0:
            raise ValueError(
                "temperature must be 0.0 when do_sample is False."
            )

    @property
    def resolved_tokenizer_name(self) -> str:
        """Kullanılacak tokenizer identifier değerini döndürür.

        tokenizer_name açıkça verilmişse onu kullanır. Aksi durumda
        tokenizer'ın model ile aynı repository'den yüklenmesi için model_name
        değerini döndürür.
        """

        if self.tokenizer_name is not None:
            return self.tokenizer_name

        return self.model_name

    def to_dict(self) -> dict[str, Any]:
        """Model config nesnesini serializable Python sözlüğüne dönüştürür."""

        return {
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any]
    ) -> "ModelConfig":
        """Python sözlüğünden doğrulanmış ModelConfig nesnesi oluşturur.

        model_name alanı zorunludur. Diğer alanlar verilmezse ModelConfig
        sınıfındaki varsayılan değerler kullanılır.

        Geçersiz veya eksik model_name değeri için ValueError oluşturur.
        """

        if not isinstance(data, dict):
            raise ValueError(
                "Model config must be a mapping."
            )

        model_name = data.get('model_name')

        if not isinstance(model_name, str):
            raise ValueError(
                "Model config must contain a string 'model_name'."
            )

        return cls(
            model_name=model_name,
            tokenizer_name=data.get("tokenizer_name"),
            revision=data.get("revision"),
            device=data.get("device", "auto"),
            dtype=data.get("dtype", "bfloat16"),
            trust_remote_code=data.get(
                "trust_remote_code",
                False
            ),
            max_new_tokens=data.get(
                "max_new_tokens",
                128
            ),
            temperature=data.get(
                "temperature",
                0.0
            ),
            do_sample=data.get(
                "do_sample",
                False
            )
        )


# @dataclass(frozen=True)
# → ModelConfig için __init__ gibi temel metodları otomatik oluşturur.
# → frozen=True config oluşturulduktan sonra değiştirilmesini engeller.
#
# Örnek:
# config = ModelConfig(model_name="Qwen/Qwen2.5-3B-Instruct")
# config.device = "cpu"  # Hata verir.


# @property
# → Bir metodu fonksiyon gibi değil, normal bir özellik gibi kullanmamızı sağlar.
#
# Örnek:
# config.resolved_tokenizer_name
# config.resolved_tokenizer_name()  # Yazmayız.


# @classmethod
# → Metodun bir nesne yerine sınıfın kendisi üzerinden çalışmasını sağlar.
# → Genellikle alternatif nesne oluşturma yöntemi için kullanılır.
#
# Örnek:
# config = ModelConfig.from_dict({
#     "model_name": "Qwen/Qwen2.5-3B-Instruct"
# })
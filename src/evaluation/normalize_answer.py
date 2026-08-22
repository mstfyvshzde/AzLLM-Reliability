"""Model prediction ve reference answer metinlerini karşılaştırma için normalize eder.

Bu modül capability evaluation öncesinde prediction ve reference answer
metinleri arasındaki yüzeysel farkları azaltır.

Amaç, anlamı değiştirmeden yalnızca karşılaştırmayı etkileyen biçimsel
farklılıkları temizlemektir.

Örnek:
    "  Four. "
    "four"

normalize sonrası:
    "four"
"""

from __future__ import annotations

import re
import string 

# unicodedata → Unicode karakterlerini standart bir forma dönüştürmek için kullanılır. Özellikle Azərbaycan dilindeki ə, ı, ş, ç, ö, ü, ğ gibi karakterlerin karşılaştırmada tutarlı temsil edilmesine yardımcı olur.
import unicodedata



def normalize_unicode(
    text: str
) -> str:
    """Metni Unicode NFC formuna dönüştürerek karakter temsilini standartlaştırır.

    Aynı görünen bazı karakterler Unicode içinde farklı biçimlerde saklanabilir.
    NFC normalization bu farklı gösterimleri mümkün olduğunca tek standart
    representation altında birleştirir.

    Örnek:
    "e" + combining accent
    → "é"

    Bu sayede özellikle Unicode karakterleri içeren EN-AZ cevap karşılaştırmaları
    daha tutarlı yapılır.
    """

    return unicodedata.normalize(
        'NFC',
        text
    )



def normalize_case(
    text: str
) -> str:
    """Metni case-insensitive karşılaştırma için lowercase forma dönüştürür."""

    return text.lower()


def remove_punctuation(
    text: str
) -> str:
    """ASCII punctuation karakterlerini metinden kaldırır.

    Örnek:
        "four."
        → "four"

        "yes!"
        → "yes"

    Harfler, rakamlar ve Azerbaijani Unicode karakterleri korunur.
    """

    translation_table = str.maketrans(
        '',
        '',
        string.punctuation
    )

    return text.translate(translation_table)


def normalize_whitespace(
    text: str
) -> str:
    """Birden fazla whitespace karakterini tek boşluğa indirger.

    Baştaki ve sondaki boşluklar da kaldırılır.

    Örnek:
        "  answer   is   four  "
        → "answer is four"
    """

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()



def normalize_answer(
    text: str
) -> str:
    """Tek bir answer string değerini karşılaştırmaya hazır hale getirir.

    Normalization sırası:

        1. Unicode NFC normalization
        2. Lowercase dönüşümü
        3. Punctuation temizleme
        4. Whitespace normalization

    Bu fonksiyon stemming, lemmatization, translation veya synonym
    replacement yapmaz.

    Böylece evaluation sırasında semantik içeriği değiştirmeden yalnızca
    yüzeysel formatting farkları azaltılır.
    """

    if not isinstance(text, str):
        raise TypeError(
            "Answer must be a string."
        )

    normalized = normalize_unicode(text)
    normalized = normalize_case(normalized)
    normalized = remove_punctuation(normalized)
    normalized = normalize_whitespace(normalized)

    return normalized
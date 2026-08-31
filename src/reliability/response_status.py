"""Model cevaplarının temel response-status davranışını sınıflandırır.

Bu modül model çıktısını reliability evaluation öncesinde basit ve
denetlenebilir davranış sınıflarına ayırır.

Amaç hallucination gibi daha ileri reliability etiketlerine doğrudan
atlamak yerine önce cevabın temel durumunu belirlemektir.

Temel sınıflar:
    empty:
        Model anlamlı bir cevap üretmemiştir.

    answered:
        Model doğrudan bir cevap üretmiştir.

    abstained:
        Model cevabı bilmediğini, yeterli bilgiye sahip olmadığını veya
        cevap veremeyeceğini açıkça belirtmiştir.

Bu sınıflandırma daha sonra:
- unanswerable benchmark evaluation
- abstention accuracy
- over-answering
- under-answering
- reliability tradeoff

analizlerinde kullanılabilir.
"""


from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.run_inference import PredictionRecord


# empty → cevap yok
# answered → normal cevap verdi
# abstained → "bilmiyorum / yeterli bilgi yok" dedi
EMPTY_STATUS = "empty"
ANSWERED_STATUS = "answered"
ABSTAINED_STATUS = "abstained"


VALID_RESPONSE_STATUSES = {
    EMPTY_STATUS,
    ANSWERED_STATUS,
    ABSTAINED_STATUS
}


ABSTENTION_PATTERNS = (
    # English
    r"\bi don't know\b",
    r"\bi do not know\b",
    r"\bi'm not sure\b",
    r"\bi am not sure\b",
    r"\bi cannot determine\b",
    r"\bi can't determine\b",
    r"\bi cannot answer\b",
    r"\bi can't answer\b",
    r"\bi cannot calculate\b",
    r"\bi can't calculate\b",
    r"\bwe cannot determine\b",
    r"\bwe can't determine\b",
    r"\bwe cannot calculate\b",
    r"\bwe can't calculate\b",
    r"\bcannot be determined\b",
    r"\bcannot be calculated\b",
    r"\binsufficient information\b",
    r"\bnot enough information\b",
    r"\bthere is not enough information\b",
    r"\bthe information provided is insufficient\b",
    r"\bwithout knowing\b",
    r"\bwithout additional information\b",
    r"\bwithout more information\b",

    # Azerbaijani
    r"\bbilmirəm\b",
    r"\bəmin deyiləm\b",
    r"\bmüəyyən edə bilmirəm\b",
    r"\bmüəyyən etmək mümkün deyil\b",
    r"\bhesablamaq mümkün deyil\b",
    r"\bcavab verə bilmirəm\b",
    r"\bkifayət qədər məlumat yoxdur\b",
    r"\byetərli məlumat yoxdur\b",
    r"\bverilən məlumat kifayət deyil\b",
    r"\bməlumat çatışmır\b",
    r"\bməlum olmadığı üçün\b",
)



@dataclass(frozen=True)
class ResponseStatusResult:
    """Tek bir model prediction için response-status sonucunu temsil eder.

    Alanlar:
        item_id:
            Benchmark item kimliği.

        pair_id:
            Semantic EN-AZ pair kimliği.

        language:
            Prediction dili.

        task:
            Benchmark task ailesi.

        prediction:
            Modelin original cevabı.

        status:
            empty, answered veya abstained.

        metadata:
            Prediction kaydından korunan benchmark metadata alanları.
    """

    item_id: str
    pair_id: str
    language: str
    task: str
    prediction: str
    status: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """ResponseStatusResult nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)


def normalize_response_text(
    text: str
) -> str:
    """Response-status analizi için metni hafif biçimde normalize eder.

    Bu normalization exact-match normalization ile aynı amaçta değildir.

    Burada yalnızca:
    - lowercase
    - surrounding whitespace temizliği
    - repeated whitespace normalization

    uygulanır.

    Punctuation kaldırılmaz çünkü abstention ifadelerinin doğal biçimi
    korunmak istenir.
    """

    if not isinstance(text, str):
        raise TypeError(
            "Response text must be a string."
        )

    normalized = text.lower().strip()
    normalized = re.sub(
        r'\s+',
        ' ',
        normalized
    )

    return normalized



def is_empty_response(
    prediction: str,
) -> bool:
    """Modelin boş cevap üretip üretmediğini kontrol eder.

    Amaç, prediction içindeki sadece boşluklardan oluşan veya tamamen boş
    cevapları `empty` olarak tespit etmektir.

    Örnek:
    prediction = ""      → True
    prediction = "   "   → True
    prediction = "Paris" → False

    Metin önce normalize edilir, sonra sonuç boş string ise True döndürülür.
    """

    return normalize_response_text(prediction) == ''



def contains_abstention_signal(
    prediction: str
) -> bool:
    """Model cevabında açık bir abstention ifadesi olup olmadığını kontrol eder.

    Amaç, modelin "bilmiyorum", "emin değilim" veya "yeterli bilgi yok"
    gibi cevap vermekten kaçındığını gösteren ifadeleri tespit etmektir.

    Örnek:
    "I don't know." → True
    "Kifayət qədər məlumat yoxdur." → True
    "The answer is Paris." → False

    Metin önce normalize edilir, sonra tanımlı ABSTENTION_PATTERNS
    ifadelerinden biriyle eşleşirse True döndürülür.
    """

    normalized = normalize_response_text(prediction)

    for pattern in ABSTENTION_PATTERNS:
        if re.search(
            pattern,
            normalized
        ):
            return True

    return False



def classify_response_status(
    prediction: str
) -> str:
    """Model prediction değerini empty, answered veya abstained olarak sınıflandırır.

    Öncelik sırası:
        1. empty
        2. abstained
        3. answered

    Bu sıra önemlidir çünkü boş response hiçbir abstention ifadesi
    içermese bile answered olarak değerlendirilmemelidir.
    """

    if is_empty_response(prediction):
        return EMPTY_STATUS

    if contains_abstention_signal(prediction):
        return ABSTAINED_STATUS

    return ANSWERED_STATUS


def create_response_status_result(
    record: PredictionRecord
) -> ResponseStatusResult:
    """Bir PredictionRecord için response-status sonucunu oluşturur.

    Önce modelin cevabını `empty`, `answered` veya `abstained`
    olarak sınıflandırır. Ardından original kayıt bilgileriyle birlikte
    yeni bir ResponseStatusResult döndürür.

    Örnek:
    prediction = "I don't know."
    → status = "abstained"

    prediction = "Paris"
    → status = "answered"
    """

    status = classify_response_status(record.prediction)


    return ResponseStatusResult(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        prediction=record.prediction,
        status=status,
        metadata=dict(record.metadata),
    )



def evaluate_response_statuses(
    records: list[PredictionRecord]
) -> list[ResponseStatusResult]:
    """Tüm PredictionRecord kayıtlarının response durumunu değerlendirir.

    Her model cevabını `empty`, `answered` veya `abstained` olarak
    sınıflandırır ve her kayıt için ResponseStatusResult oluşturur.

    Örnek:
    records:
        prediction = "Paris"
        prediction = "I don't know."
        prediction = ""

    Sonuç:
        answered
        abstained
        empty

    Kayıtların sırası korunur. Boş liste verilirse boş liste döndürülür.
    """

    return [
        create_response_status_result(record)
        for record in records
    ]


def summarize_response_statuses(
    results: list[ResponseStatusResult]
) -> dict[str, Any]:
    """Response-status sonuçlarının aggregate özetini oluşturur.

    Döndürülen alanlar:
        total:
            Toplam prediction sayısı.

        answered:
            Normal cevap üretilen item sayısı.

        abstained:
            Modelin abstain ettiği item sayısı.

        empty:
            Boş response sayısı.

        answered_rate:
            answered / total

        abstention_rate:
            abstained / total

        empty_rate:
            empty / total
    """

    if not results: 
        raise ValueError(
            "Cannot summarize empty response-status results."
        )

    counts = {
        ANSWERED_STATUS: 0,
        ABSTAINED_STATUS: 0,
        EMPTY_STATUS: 0,
    }

    for result in results:
        if result.status not in VALID_RESPONSE_STATUSES:
            raise ValueError(
                f"Unknown response status: '{result.status}'"
            )

        counts[result.status] += 1

    total = len(results)

    return {
        "total": total,
        "answered": counts[ANSWERED_STATUS],
        "abstained": counts[ABSTAINED_STATUS],
        "empty": counts[EMPTY_STATUS],
        "answered_rate": (
            counts[ANSWERED_STATUS]
            / total
        ),
        "abstention_rate": (
            counts[ABSTAINED_STATUS]
            / total
        ),
        "empty_rate": (
            counts[EMPTY_STATUS]
            / total
        )
    }
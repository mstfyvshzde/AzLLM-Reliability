"""Benchmark kayıtlarının standart veri yapısını tanımlar.

Bu modül, benchmark içindeki her bir sorunun aynı formatta tutulmasını sağlar.
İngilizce ve Azerbaycanca eşdeğer örnekler `pair_id` ile birbirine bağlanır.

Her kayıt şu temel bilgileri içerir:
- benzersiz örnek kimliği
- eşleşme kimliği
- dil
- soru
- referans cevap
- isteğe bağlı metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any




# frozen=True → oluşturulan kaydı değiştirilemez yapar.
@dataclass(frozen=True)
class BenchmarkRecord:
    """Tek bir benchmark örneğini temsil eder.

    Attributes:
        item_id: Benchmark içindeki benzersiz örnek kimliği.
        pair_id: İngilizce ve Azerbaycanca eşdeğer örnekleri bağlayan kimlik.
        language: Örneğin dili.
        question: Modele yöneltilecek soru veya görev.
        reference_answer: Değerlendirmede kullanılacak referans cevap.
        metadata: Kaynak veya analiz için isteğe bağlı ek bilgiler.
    """

    item_id: str
    pair_id: str
    language: str
    task: str
    question: str
    reference_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Kaydı serileştirilebilir bir sözlüğe dönüştürür."""

        return {
            "item_id": self.item_id,
            "pair_id": self.pair_id,
            "language": self.language,
            "task": self.task,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "metadata": self.metadata,
        }




# pair_id: reasoning_001

# EN:
# item_id: reasoning_001_en
# question: Ali has 3 books and buys 2 more. How many books does he have?
# answer: 5

# AZ:
# item_id: reasoning_001_az
# question: Əlinin 3 kitabı var və o, daha 2 kitab alır. İndi onun neçə kitabı var?
# answer: 5
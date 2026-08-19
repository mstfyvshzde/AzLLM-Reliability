"""Benchmark kayıtlarının doğru ve kullanılabilir olup olmadığını kontrol eder.

Her kayıt için item_id, pair_id, language, question ve reference_answer
alanlarını kontrol eder. Boş alanları, desteklenmeyen dilleri, tekrar eden
item_id değerlerini ve aynı pair_id-language kombinasyonunun birden fazla
kullanılmasını tespit eder.
"""


from __future__ import annotations

# Bir fonksiyonun liste, tuple, set gibi üzerinde tek tek dolaşılabilen verileri kabul ettiğini belirtmek için kullanılır.
from collections.abc import Iterable

# BenchmarkRecord buraya gelen itemlerde şunları getirir:
# item_id, pair_id, language, questionm, reference_answer, metadata
from src.data.benchmark_record import BenchmarkRecord


def validate_record(
    record: BenchmarkRecord,
    allowed_languages: set[str],
    reject_empty_questions: bool = True,
    reject_empty_answers: bool = True
) ->None:
    """Tek bir benchmark kaydını doğrular.

    Args:
        record: Doğrulanacak benchmark kaydı.
        allowed_languages: Benchmark içinde kabul edilen dil kodları.

    Raises:
        ValueError: Kayıt gerekli alanları veya dil kurallarını ihlal ederse.
    """

    if not record.item_id.strip():
        return ValueError("item_id cannot be empty.")

    if not record.pair_id.strip():
        raise ValueError("pair_id cannot be empty.")

    if record.language not in allowed_languages:
        raise ValueError(
            f"Unsupported language '{record.language}'. "
            f"Expected one of: {sorted(allowed_languages)}"
        )

    if reject_empty_questions and not record.question.strip():
        raise ValueError(
            f"Question cannot be empty for item '{record.item_id}'."
        )

    if reject_empty_answers and not record.reference_answer.strip():
        raise ValueError(
            f"Reference answer cannot be empty for item '{record.item_id}'."
        )


def validate_unique_item_ids(
    records: Iterable[BenchmarkRecord]
) -> None:
    """Benchmark kayıtlarında tekrar eden item_id değerlerini kontrol eder.

    Args:
        records: Doğrulanacak benchmark kayıtları.

    Raises:
        ValueError: Aynı item_id birden fazla kez kullanılmışsa.
    """

    # Daha önce gördüğümüz item_id değerlerini saklayan boş bir küme (set) oluşturur.
    seen_ids: set[str] = set()

    for record in records:
        if record.item_id in seen_ids:
            raise ValueError(
                f"Duplicate item_id detected: {record.item_id}"
            )

        seen_ids.add(record.item_id)



def validate_unique_pair_languages(
    records: Iterable[BenchmarkRecord]
) -> None:
    """Aynı pair_id ve language kombinasyonunun tekrarını kontrol eder.

    Args:
        records: Doğrulanacak benchmark kayıtları.

    Raises:
        ValueError: Aynı pair_id-language kombinasyonu birden fazla kez varsa.
    """

    seen_pairs: set[tuple[str, str]] = set()

    for record in records:
        key = (record.pair_id, record.language)

        # pair_id: reasoning_001
        # language: en

        # pair_id: reasoning_001
        # language: az
        if key in seen_pairs:
            raise ValueError(
                "Duplicate pair-language combination detected: "
                f"pair_id={record.pair_id}, language={record.language}"
            )

        seen_pairs.add(key)


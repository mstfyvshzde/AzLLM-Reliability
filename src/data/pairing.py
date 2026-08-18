"""İngilizce-Azerbaycanca benchmark eşleşmelerinin bütünlüğünü kontrol eder.

Aynı pair_id'ye sahip kayıtları gruplar ve her pair içinde gerekli tüm
dillerin (örneğin "en" ve "az") bulunup bulunmadığını doğrular.

Eksik (missing) veya beklenmeyen (unexpected) bir dil varsa hata verir.
"""

from __future__ import annotations

# Olmayan bir key'e otomatik başlangıç değeri verir.
from collections import defaultdict
from collections.abc import Iterable

from src.data.benchmark_record import BenchmarkRecord



def group_by_pair(
    records: Iterable[BenchmarkRecord]
) -> dict[str, list[BenchmarkRecord]]:
    """Benchmark kayıtlarını pair_id değerine göre gruplar.
    
    Dostum örneğin elimizde 4 kayıt var:
    reasoning_001_en → pair_id: reasoning_001
    reasoning_001_az → pair_id: reasoning_001
    reasoning_002_en → pair_id: reasoning_002
    reasoning_002_az → pair_id: reasoning_002

    group_by_pair() bunları şöyle gruplar:
    reasoning_001 → [reasoning_001_en, reasoning_001_az]
    reasoning_002 → [reasoning_002_en, reasoning_002_az]"""

    pairs: dict[str, list[BenchmarkRecord]] = defaultdict(list)

    for record in records:
        pairs[record.pair_id].append(record)

    return dict(pairs)



def validate_complete_pairs(
    records: Iterable[BenchmarkRecord],
    required_languages: set[str]
) -> None:
    """Her pair'in gerekli tüm dil sürümlerini içerdiğini doğrular."""

    pairs = group_by_pair(records)

    for pair_id, pair_records in pairs.items():
        languages = {record.language for record in pair_records}

        # Gerekli diller: {"en", "az"}
        # Missing:
        # Var: {"en"}
        # Eksik: {"az"}  → missing
        # Unexpected:
        # Var: {"en", "az", "tr"}
        # Fazladan: {"tr"} → unexpected
        if languages != required_languages:
            missing = required_languages - languages
            unexpected = languages - required_languages

            raise ValueError(
                f"Incomplete pair '{pair_id}'. "
                f"Missing languages: {sorted(missing)}. "
                f"Unexpected languages: {sorted(unexpected)}."
            )
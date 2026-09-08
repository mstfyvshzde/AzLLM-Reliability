"""Adaptation training kayıt yapısını tanımlar.

Bu modül, Azerbaijani adaptation datası için kullanılan tekil
training example formatını merkezi biçimde tanımlar.

Her kayıt benchmark'tan bağımsız üretilmelidir ve frozen benchmark
içeriğinin doğrudan veya semantik kopyasını içermemelidir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AdaptationRecord:
    """Tek bir adaptation training örneğini temsil eder."""

    item_id: str
    language: str
    category: str
    instruction: str
    response: str
    source: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Record'u JSON-uyumlu dictionary olarak döndürür."""
        return asdict(self)

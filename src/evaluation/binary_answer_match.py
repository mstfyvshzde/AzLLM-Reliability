"""Binary Yes/No cevaplarını değerlendirir.

Politika:

- Reference yalnızca:
    Yes
    No
    Bəli
    Xeyr

- Prediction açık binary polarity içeriyorsa deterministic değerlendirilir.
- Prediction açık polarity içermiyorsa human adjudication decision gerekir.
- Item-specific kural kullanılmaz.
"""

from __future__ import annotations

import re

from src.evaluation.normalize_answer import normalize_answer
from src.evaluation.run_inference import PredictionRecord


POSITIVE_REFERENCES = {
    "yes",
    "bəli",
}

NEGATIVE_REFERENCES = {
    "no",
    "xeyr",
}

BINARY_REFERENCES = (
    POSITIVE_REFERENCES
    | NEGATIVE_REFERENCES
)


def normalize_binary_reference(
    reference_answer: str,
) -> str:
    """Reference cevabı binary canonical forma dönüştürür."""

    normalized = normalize_answer(
        reference_answer
    )

    if normalized in POSITIVE_REFERENCES:
        return "positive"

    if normalized in NEGATIVE_REFERENCES:
        return "negative"

    raise ValueError(
        "Reference answer is not binary: "
        f"{reference_answer!r}"
    )


def detect_explicit_binary_polarity(
    prediction: str,
) -> str | None:
    """Prediction içindeki açık Yes/No/Bəli/Xeyr polarity'sini bulur."""

    normalized = normalize_answer(
        prediction
    )

    positive_patterns = (
        r"\byes\b",
        r"\bbəli\b",
    )

    negative_patterns = (
        r"\bno\b",
        r"\bxeyr\b",
    )

    has_positive = any(
        re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )
        for pattern in positive_patterns
    )

    has_negative = any(
        re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )
        for pattern in negative_patterns
    )

    if has_positive and has_negative:
        return None

    if has_positive:
        return "positive"

    if has_negative:
        return "negative"

    return None


def binary_answer_match_score(
    record: PredictionRecord,
    adjudication_decisions: dict[str, int] | None = None,
) -> int:
    """Binary prediction için correctness score üretir."""

    reference_polarity = normalize_binary_reference(
        record.reference_answer
    )

    explicit_polarity = detect_explicit_binary_polarity(
        record.prediction
    )

    if explicit_polarity is not None:
        return int(
            explicit_polarity
            == reference_polarity
        )

    if adjudication_decisions is None:
        raise ValueError(
            "Binary adjudication decisions are required "
            f"for item: {record.item_id}"
        )

    if record.item_id not in adjudication_decisions:
        raise ValueError(
            "Missing binary adjudication decision: "
            f"{record.item_id}"
        )

    decision = adjudication_decisions[
        record.item_id
    ]

    if decision not in {0, 1}:
        raise ValueError(
            "Binary adjudication decision must be 0 or 1."
        )

    return decision


def is_binary_reference(
    reference_answer: str,
) -> bool:
    """Reference binary Yes/No cevabı mı kontrol eder."""

    return (
        normalize_answer(reference_answer)
        in BINARY_REFERENCES
    )

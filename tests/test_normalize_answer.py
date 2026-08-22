"""Answer normalization yardımcı fonksiyonlarını test eder."""

import pytest

from src.evaluation.normalize_answer import (
    normalize_answer,
    normalize_case,
    normalize_unicode,
    normalize_whitespace,
    remove_punctuation,
)


def test_normalize_unicode() -> None:
    """Unicode metnin NFC formuna normalize edildiğini test eder."""

    decomposed = "A\u0301"

    normalized = normalize_unicode(
        decomposed
    )

    assert normalized == "Á"


def test_normalize_case() -> None:
    """Metnin lowercase forma dönüştürüldüğünü test eder."""

    assert normalize_case(
        "FOUR"
    ) == "four"


def test_remove_punctuation() -> None:
    """ASCII punctuation karakterlerinin kaldırıldığını test eder."""

    assert remove_punctuation(
        "four."
    ) == "four"

    assert remove_punctuation(
        "yes!"
    ) == "yes"


def test_remove_punctuation_preserves_azerbaijani_characters() -> None:
    """Azerbaijani Unicode karakterlerinin punctuation temizliğinde korunduğunu test eder."""

    result = remove_punctuation(
        "Azərbaycan, yaxşıdır!"
    )

    assert result == "Azərbaycan yaxşıdır"


def test_normalize_whitespace() -> None:
    """Fazla whitespace karakterlerinin tek boşluğa indirildiğini test eder."""

    result = normalize_whitespace(
        "  answer   is   four  "
    )

    assert result == "answer is four"


def test_normalize_whitespace_handles_tabs_and_newlines() -> None:
    """Tab ve newline karakterlerinin normal boşluğa dönüştürüldüğünü test eder."""

    result = normalize_whitespace(
        "answer\tis\nfour"
    )

    assert result == "answer is four"


def test_normalize_answer() -> None:
    """Tüm normalization adımlarının birlikte çalıştığını test eder."""

    result = normalize_answer(
        "  Four. "
    )

    assert result == "four"


def test_normalize_answer_with_azerbaijani_text() -> None:
    """Azerbaijani metnin karakterleri korunarak normalize edildiğini test eder."""

    result = normalize_answer(
        "  AZƏRBAYCAN!  "
    )

    assert result == "azərbaycan"


def test_normalize_answer_preserves_numbers() -> None:
    """Sayısal answer değerlerinin normalization sırasında korunduğunu test eder."""

    result = normalize_answer(
        "  42. "
    )

    assert result == "42"


def test_normalize_answer_empty_string() -> None:
    """Boş string değerinin boş normalized answer olarak döndüğünü test eder."""

    assert normalize_answer(
        ""
    ) == ""


def test_normalize_answer_whitespace_only() -> None:
    """Yalnızca whitespace içeren cevabın boş string'e dönüştüğünü test eder."""

    assert normalize_answer(
        "   \n\t  "
    ) == ""


def test_normalize_answer_rejects_non_string() -> None:
    """String olmayan answer değerinin reddedildiğini test eder."""

    with pytest.raises(
        TypeError,
        match="Answer must be a string",
    ):
        normalize_answer(
            42  # type: ignore[arg-type]
        )


def test_semantically_equal_surface_forms_match() -> None:
    """Yüzeysel biçimi farklı fakat aynı answer değerlerinin eşitlendiğini test eder."""

    first = normalize_answer(
        "Four."
    )

    second = normalize_answer(
        "  four  "
    )

    assert first == second
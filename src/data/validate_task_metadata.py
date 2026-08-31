"""Task-specific benchmark metadata kurallarını doğrular.

Bu modül benchmark kayıtlarındaki category, difficulty ve review_status
alanlarının ilgili task specification ile uyumlu olup olmadığını kontrol eder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.data.benchmark_record import BenchmarkRecord


def load_task_specifications(specifications_path: Path) -> dict[str, Any]:
    """Task specification YAML dosyasını yükler ve temel yapısını doğrular.

    Specifications, bir task için hangi category, difficulty ve metadata
    kurallarının geçerli olduğunu tanımlar.

    Örnek:
        specifications_path = Path("configs/task_specs/reasoning.yaml")

        YAML içeriği:
        task:
        name: "reasoning"

        difficulty:
        levels: ["easy", "medium", "hard"]

    Fonksiyon YAML içeriğini Python sözlüğüne dönüştürür.
    Dosya bulunamazsa FileNotFoundError, içerik geçerli bir mapping değilse
    veya `task` bölümü yoksa ValueError oluşturur.
    """

    if not specifications_path.exists():
        raise FileNotFoundError(
            f"Task specification not found: {specifications_path}"
        )

    with specifications_path.open('r', encoding='utf-8') as file:
        specifications = yaml.safe_load(file)

    if not isinstance(specifications, dict):
        raise ValueError(
            "Task specification must contain a YAML mapping."
        )

    if 'task' not in specifications:
        raise ValueError(
            "Task specification must contain a 'task' section."
        )

    return specifications




def validate_required_metadata(
    record: BenchmarkRecord,
    specifications: dict[str, Any]
) -> None:
    """Task specification tarafından zorunlu tutulan metadata alanlarını doğrular.

    Specification içindeki `required_fields` listesini alır ve her alanın
    BenchmarkRecord.metadata içinde mevcut ve boş olmadığını kontrol eder.

    Örnek required metadata fields:
        category
        difficulty
        review_status

    Örnek:
        metadata = {
            "category": "logical_reasoning",
            "difficulty": "medium",
            "review_status": "approved",
        }

    Bu alanlardan biri eksik veya boşsa ValueError oluşturur.
    """

    required_fields = specifications.get(
        'metadata',
        {}
    ).get(
        'required_fields',
        []
    )

    for field in required_fields:
        value = record.metadata.get(field)

        if value is None or (
            isinstance(value, str) and not value.strip()
        ):
            raise ValueError(
                f"Missing required metadata '{field}' "
                f"for item '{record.item_id}'."
            )


def validate_category(
    record: BenchmarkRecord,
    specifications: dict[str, Any]
) -> None:
    """Kaydın category değerinin task specification içinde izin verilen bir kategori olduğunu doğrular.

    Specification içindeki `categories` bölümünden yalnızca `enabled: true`
    olan kategorileri toplar ve kaydın metadata içindeki `category` değerini
    bunlarla karşılaştırır.

    Örnek specification:
        categories:
            logical_reasoning:
            enabled: true
        arithmetic:
            enabled: true
        translation:
            enabled: false

    Örnek record metadata:
        category = "logical_reasoning"

    Bu geçerlidir.

    Ama:
        category = "translation"

    enabled olmadığı için ValueError oluşturur.
    """

    categories = specifications.get('categories', {})

    enabled_categories = {
        category
        for category, settings in categories.items()
        if isinstance(settings, dict)
        and settings.get('enabled', False)
    }

    category = record.metadata.get('category')

    if category not in enabled_categories:
        raise ValueError(
            f"Unsupported category '{category}' "
            f"for task '{record.task}'."
        )



def validate_difficulty(
    record: BenchmarkRecord,
    specifications: dict[str, Any]
) -> None:
    """Kaydın difficulty seviyesinin task specification içinde izin verilen bir seviye olduğunu doğrular.

    Specification içindeki `difficulty.levels` listesini alır ve kaydın
    metadata içindeki `difficulty` değerini bu seviyelerle karşılaştırır.

    Örnek specification:
        difficulty:
            levels: ["easy", "medium", "hard"]

    Örnek record metadata:
        difficulty = "medium"

    Bu geçerlidir.

    Ama:
        difficulty = "expert"

    izin verilen seviyeler arasında olmadığı için ValueError oluşturur.
    """

    allowed_levels = set(
        specifications.get('difficulty', {}).get('levels', [])
    )

    difficulty = record.metadata.get('difficulty')

    if difficulty not in allowed_levels:
        raise ValueError(
            f"Unsupported difficulty '{difficulty}' "
            f"for item '{record.item_id}'."
        )



def validate_metadata_constraints(
    record: BenchmarkRecord,
    specifications: dict[str, Any],
) -> None:
    """Task specification içindeki sabit metadata constraint'lerini doğrular.

    Örnek:
        metadata:
            constraints:
                is_answerable: true

    Bu durumda record.metadata içindeki is_answerable değeri mutlaka True
    olmalıdır.
    """

    constraints = specifications.get(
        "metadata",
        {},
    ).get(
        "constraints",
        {}
    )

    if not isinstance(
        constraints,
        dict
    ):
        raise ValueError(
            "metadata.constraints must be a mapping."
        )

    for field, expected_value in constraints.items():
        if field not in record.metadata:
            raise ValueError(
                f"Missing constrained metadata '{field}' "
                f"for item '{record.item_id}'."
            )

        actual_value = record.metadata[
            field
        ]

        if actual_value != expected_value:
            raise ValueError(
                f"Invalid metadata constraint '{field}' "
                f"for item '{record.item_id}': "
                f"expected {expected_value!r}, got {actual_value!r}."
            )


def validate_task_metadata(
    record: BenchmarkRecord,
    specifications: dict[str, Any]
) -> None:
    """Bir benchmark kaydının task-specific metadata bütünlüğünü doğrular.

    `record`, doğrulanacak tek benchmark kaydıdır.
    `specifications` ise o task için geçerli kuralları içerir.

    Örnek record:
        task = "reasoning"
        metadata = {
            "category": "logical_reasoning",
            "difficulty": "medium",
            "review_status": "approved",
        }

    Örnek specifications:
        task:
            name: "reasoning"

        categories:
            logical_reasoning:
            enabled: true

        difficulty:
            levels: ["easy", "medium", "hard"]

        metadata:
            required_fields:
            - category
            - difficulty
            - review_status

    Fonksiyon önce record.task ile specification içindeki task adının eşleştiğini
    kontrol eder. Ardından category, required metadata ve difficulty kurallarını
    doğrular.

    Kurallardan biri ihlal edilirse ValueError oluşturur.
    """

    expected_task = specifications[
        "task"
    ][
        "name"
    ]

    if record.task != expected_task:
        raise ValueError(
            f"Record task '{record.task}' does not match "
            f"task specification '{expected_task}'."
        )

    validate_category(
        record,
        specifications
    )

    validate_required_metadata(
        record,
        specifications
    )

    validate_difficulty(
        record,
        specifications
    )

    validate_metadata_constraints(
        record,
        specifications
    )
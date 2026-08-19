"""Benchmark görev tanımlarını yükler ve doğrular.

Bu modül, configs/tasks.yaml dosyasındaki task family tanımlarını
okur ve benchmark kayıtlarında kullanılan task değerlerinin
geçerli olup olmadığını kontrol eder.
"""


from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_task_config(config_path: Path) -> dict[str, Any]:
    """Task yapılandırma dosyasını yükler.

    `configs/tasks.yaml` dosyasını açar, YAML içeriğini Python sözlüğüne
    dönüştürür ve geri döndürür.

    Dosya bulunamazsa FileNotFoundError, içerik geçerli bir mapping değilse
    ValueError oluşturur.

    Örnek yapılandırma dosyası:
    tasks:
        reasoning:
            enabled: true

        factual_qa:
            enabled: true

        translation:
            enabled: false
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")

    with config_path.open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Task config must contain a YAML mapping.")

    if 'tasks' not in config:
        raise ValueError("Task config must contain a 'tasks' section.")

    if not isinstance(config['tasks'], dict):
        raise ValueError("'tasks' section must be a YAML mapping.")

    return config



def get_enabled_tasks(config: dict[str, Any]) -> set[str]:
    """Config içindeki aktif benchmark task'larını bulur.

    `tasks` bölümündeki task'ları tek tek kontrol eder.
    `enabled: true` olan task isimlerini bir set içinde toplar ve döndürür.

    Örnek:
    reasoning → true
    factual_qa → true
    translation → false

    Sonuç:
    {"reasoning", "factual_qa"}
    """

    enabled_tasks: set[str] = set()

    for task_name, task_config in config['tasks'].items():
        if not isinstance(task_config, dict):
            raise ValueError(
                f"Task configuration must be a mapping: {task_name}"
            )

        if task_config.get('enabled', False):
            enabled_tasks.add(task_name)

    return enabled_tasks




def validate_task(
    task: str,
    enabled_tasks: set[str]
) -> None:
    """Benchmark kaydındaki task değerini doğrular."""
    if not task.strip():
        raise ValueError("Task cannot be empty.")

    if task not in enabled_tasks:
        raise ValueError(
            f"Unsupported or disabled task '{task}'. "
            f"Expected one of: {sorted(enabled_tasks)}"
        )
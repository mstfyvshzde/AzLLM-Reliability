"""Task registry yükleme ve doğrulama davranışlarını test eder."""

from pathlib import Path

import pytest

from src.data.task_registry import (
    get_enabled_tasks,
    load_task_config,
    validate_task,
)


def test_load_task_config(tmp_path: Path) -> None:
    """Geçerli task config dosyasının doğru yüklendiğini test eder."""
    config_path = tmp_path / "tasks.yaml"

    config_path.write_text(
        """
tasks:
  reasoning:
    enabled: true
    answer_type: short_answer
""",
        encoding="utf-8",
    )

    config = load_task_config(config_path)

    assert "tasks" in config
    assert "reasoning" in config["tasks"]


def test_missing_tasks_section_is_rejected(tmp_path: Path) -> None:
    """tasks bölümü olmayan config dosyasının reddedildiğini test eder."""
    config_path = tmp_path / "tasks.yaml"

    config_path.write_text(
        "project: test\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="tasks"):
        load_task_config(config_path)


def test_enabled_tasks_are_returned() -> None:
    """Yalnızca enabled=true olan task'ların döndürüldüğünü test eder."""
    config = {
        "tasks": {
            "reasoning": {"enabled": True},
            "factual_knowledge": {"enabled": True},
            "unanswerable": {"enabled": False}
        }
    }

    enabled_tasks = get_enabled_tasks(config)

    assert enabled_tasks == {
        "reasoning",
        "factual_knowledge"
    }


def test_valid_task() -> None:
    """Etkin bir task değerinin doğrulamadan geçtiğini test eder."""
    validate_task(
        "reasoning",
        {"reasoning", "factual_knowledge"}
    )


def test_disabled_or_unknown_task_is_rejected() -> None:
    """Etkin olmayan veya bilinmeyen task değerinin reddedildiğini test eder."""
    with pytest.raises(ValueError, match="Unsupported or disabled task"):
        validate_task(
            "translation",
            {"reasoning", "factual_knowledge"}
        )


def test_empty_task_is_rejected() -> None:
    """Boş task değerinin reddedildiğini test eder."""
    with pytest.raises(ValueError, match="Task cannot be empty"):
        validate_task(
            "",
            {"reasoning"}
        )
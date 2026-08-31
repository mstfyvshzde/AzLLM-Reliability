"""Baseline experiment config yapısının gerekli alanlarını ve temel değerlerini doğrular.

Kontrol edilen bölümler:
    - experiment
    - languages
    - model
    - benchmark
    - inference
    - evaluation
    - artifacts

Ayrıca:
    - experiment name boş mu?
    - seed integer mı?
    - source ve target language tanımlı ve birbirinden farklı mı?
    - model config_path var mı?
    - benchmark input_path var mı?
    - inference output_path var mı?
    - evaluation output_dir var mı?

Eksik veya geçersiz bir config alanı bulunursa ValueError veya TypeError oluşturur.
"""

from __future__ import annotations

from typing import Any


def validate_experiment_config(
    config:dict[str, Any]
) -> None:
    """Baseline experiment config yapısının gerekli alanlarını doğrular."""

    required_sections = {
        "experiment",
        "languages",
        "model",
        "benchmark",
        "inference",
        "evaluation",
        "artifacts"
    }
    

    missing_sections = (
        required_sections
        - set(config)
    )

    if missing_sections:
        missing = ", ".join(
            sorted(missing_sections)
        )

        raise ValueError(
            f"Missing experiment config sections: {missing}"
        )

    experiment = config[
        "experiment"
    ]

    if not isinstance(
        experiment,
        dict,
    ):
        raise ValueError(
            "'experiment' section must be a mapping."
        )

    if not experiment.get(
        "name"
    ):
        raise ValueError(
            "Experiment name cannot be empty."
        )

    seed = experiment.get(
        "seed"
    )

    if not isinstance(
        seed,
        int,
    ):
        raise TypeError(
            "Experiment seed must be an integer."
        )

    languages = config[
        "languages"
    ]

    if not isinstance(
        languages,
        dict,
    ):
        raise ValueError(
            "'languages' section must be a mapping."
        )

    source_language = languages.get(
        "source"
    )

    target_language = languages.get(
        "target"
    )

    if not source_language:
        raise ValueError(
            "Source language cannot be empty."
        )

    if not target_language:
        raise ValueError(
            "Target language cannot be empty."
        )

    if source_language == target_language:
        raise ValueError(
            "Source and target languages must be different."
        )

    model = config[
        "model"
    ]

    if not isinstance(
        model,
        dict,
    ):
        raise ValueError(
            "'model' section must be a mapping."
        )

    if not model.get(
        "config_path"
    ):
        raise ValueError(
            "Model config_path cannot be empty."
        )

    benchmark = config[
        "benchmark"
    ]

    if not isinstance(
        benchmark,
        dict,
    ):
        raise ValueError(
            "'benchmark' section must be a mapping."
        )

    if not benchmark.get(
        "input_path"
    ):
        raise ValueError(
            "Benchmark input_path cannot be empty."
        )

    inference = config[
        "inference"
    ]

    if not isinstance(
        inference,
        dict,
    ):
        raise ValueError(
            "'inference' section must be a mapping."
        )

    if not inference.get(
        "output_path"
    ):
        raise ValueError(
            "Inference output_path cannot be empty."
        )

    evaluation = config[
        "evaluation"
    ]

    if not isinstance(
        evaluation,
        dict,
    ):
        raise ValueError(
            "'evaluation' section must be a mapping."
        )

    if not evaluation.get(
        "output_dir"
    ):
        raise ValueError(
            "Evaluation output_dir cannot be empty."
        )
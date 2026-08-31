"""Baseline experiment başlamadan önce gerekli preflight kontrollerini çalıştırır.

Bu modül gerçek model yükleme ve inference başlamadan önce experiment ortamının
hazır olup olmadığını kontrol eder.

Kontroller:
- experiment config geçerli mi?
- model config dosyası mevcut mu?
- benchmark input dosyası mevcut mu?
- benchmark boş mu?
- source ve target language pair'leri complete mi?
- model identifier geçerli mi?
- output artifact path'leri mevcut run ile çakışıyor mu?
- overwrite kapalıysa eski artifact var mı?

Amaç pahalı model yükleme veya inference başlamadan önce mümkün olan hataları
erken ve açık biçimde yakalamaktır.
"""



from __future__ import annotations

import argparse

from dataclasses import asdict, dataclass
from typing import Any
from pathlib import Path

from src.data.build_benchmark import load_records
from src.data.pairing import (
    validate_complete_pairs,
    validate_pair_task_consistency
)
from src.experiments.config_validation import (
    validate_experiment_config,
)
from src.models.load_config import load_model_config



@dataclass(frozen=True)
class PreflightResult:
    """Baseline preflight kontrol sonucunu temsil eder.

    Alanlar:
        experiment_name:
            Kontrol edilen experiment adı.

        benchmark_record_count:
            Benchmark input içindeki toplam record sayısı.

        pair_count:
            Benchmark içindeki unique semantic pair sayısı.

        source_language:
            Source language kodu.

        target_language:
            Target language kodu.

        model_name:
            Kullanılacak base model identifier.

        ready:
            Tüm preflight kontrolleri geçtiyse True.
    """

    experiment_name: str
    benchmark_record_count: int
    pair_count: int
    source_language: str
    target_language: str
    model_name: str
    ready: bool

    def to_dict(self) -> dict[str, Any]:
        """PreflightResult nesnesini serializable dictionary'ye dönüştürür."""

        return asdict(self)



def validate_existing_output(
    output_path: Path,
    overwrite: bool,
    artifact_name: str
) -> None:
    """Mevcut output artifact'ının üzerine yanlışlıkla yazılmasını engeller.

    `output_path` zaten varsa ve `overwrite=False` ise FileExistsError oluşturur.

    Örnek:
    output_path = "outputs/predictions.jsonl"
    overwrite = False

    Dosya zaten varsa:
    → FileExistsError

    `overwrite=True` ise mevcut dosyanın üzerine yazılmasına izin verilir.
    """

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{artifact_name} already exists: {output_path}"
        )



def validate_output_paths(
    config: dict[str, Any]
) -> None:
    """Baseline experiment output path'lerinde çakışma olup olmadığını kontrol eder.

    Prediction, evaluation, config snapshot ve run metadata çıktılarının mevcut
    dosyalarla çakışıp çakışmadığını kontrol eder.

    `config snapshot`, experiment sırasında kullanılan config ayarlarının o run'a
    ait kopyasıdır.

    Örnek:
    experiment_config.yaml
    → model, seed, language, benchmark ve diğer run ayarlarını saklar.

    `overwrite=False` iken ilgili çıktı zaten mevcutsa FileExistsError oluşturur.
    """

    inference = config[
        "inference"
    ]

    evaluation = config[
        "evaluation"
    ]

    artifacts = config[
        "artifacts"
    ]

    prediction_path = Path(
        inference[
            "output_path"
        ]
    )

    evaluation_dir = Path(
        evaluation[
            "output_dir"
        ]
    )

    inference_overwrite = inference.get(
        "overwrite",
        False
    )

    evaluation_overwrite = evaluation.get(
        "overwrite",
        False
    )

    validate_existing_output(
        output_path=prediction_path,
        overwrite=inference_overwrite,
        artifact_name="Prediction artifact"
    )

    if evaluation_dir.exists():
        existing_files = [
            path
            for path in evaluation_dir.iterdir()
            if path.is_file()
        ]

        if existing_files and not evaluation_overwrite:
            raise FileExistsError(
                f"Evaluation output directory is not empty: {evaluation_dir}"
            )

    if artifacts.get(
        "save_config_snapshot",
        False
    ):
        config_snapshot_path = (
            prediction_path.parent
            / "experiment_config.yaml"
        )

        validate_existing_output(
            output_path=config_snapshot_path,
            overwrite=inference_overwrite,
            artifact_name="Config snapshot"
        )

    if artifacts.get(
        "save_run_metadata",
        False
    ):
        metadata_path = (
            prediction_path.parent
            / "run_metadata.json"
        )

        validate_existing_output(
            output_path=metadata_path,
            overwrite=inference_overwrite,
            artifact_name="Run metadata"
        )



def validate_benchmark_pairs(
    records: list[Any],
    source_language: str,
    target_language: str,
    require_complete_pairs: bool
) -> int:
    """Benchmark kayıtlarının EN-AZ pair bütünlüğünü doğrular ve pair sayısını döndürür.

    Kontroller:
    - Benchmark boş mu?
    - Aynı pair içindeki task değerleri tutarlı mı?
    - require_complete_pairs=True ise her pair hem source hem target dili içeriyor mu?

    Örnek:
    reasoning_001_en → pair_id="reasoning_001"
    reasoning_001_az → pair_id="reasoning_001"

    reasoning_002_en → pair_id="reasoning_002"
    reasoning_002_az → pair_id="reasoning_002"

    Sonuç:
    pair_count = 2

    Kurallardan biri bozulursa ValueError oluşturur.
    """

    if not records:
        raise ValueError(
            "Benchmark input contains no records."
        )

    validate_pair_task_consistency(records)

    if require_complete_pairs:
        validate_complete_pairs(
            records,
            {
                source_language,
                target_language
            }
        )

    pair_ids = {
        record.pair_id
        for record in records
    }

    return len(pair_ids)


def run_preflight(
    experiment_config: dict[str, Any]
) -> PreflightResult:
    """Baseline experiment başlamadan önce tüm preflight kontrollerini çalıştırır.

    Akış:
    1. Experiment config doğrulanır.
    2. Source ve target language ayarları alınır.
    3. Model config ve benchmark path'leri hazırlanır.
    4. Gerekli dosyaların mevcut olduğu kontrol edilir.
    5. Model config yüklenir ve model_name doğrulanır.
    6. Benchmark kayıtları yüklenir.
    7. EN-AZ pair bütünlüğü ve pair sayısı kontrol edilir.
    8. Output path çakışmaları kontrol edilir.
    9. Tüm kontroller geçerse ready=True olan PreflightResult döndürülür.

    Örnek sonuç:
    PreflightResult(
        experiment_name="baseline_qwen",
        benchmark_record_count=200,
        pair_count=100,
        source_language="en",
        target_language="az",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        ready=True
    )
    """

    validate_experiment_config(
        experiment_config
    )

    experiment = experiment_config[
        "experiment"
    ]

    languages = experiment_config[
        "languages"
    ]

    model_section = experiment_config[
        "model"
    ]

    benchmark = experiment_config[
        "benchmark"
    ]

    source_language = languages[
        "source"
    ]

    target_language = languages[
        "target"
    ]

    model_config_path = Path(
        model_section[
            "config_path"
        ]
    )

    benchmark_path = Path(
        benchmark[
            "input_path"
        ]
    )

    if not model_config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {model_config_path}"
        )

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark input not found: {benchmark_path}"
        )

    model_config = load_model_config(
        model_config_path
    )

    if not model_config.model_name.strip():
        raise ValueError(
            "Model name cannot be empty."
        )

    records = load_records(
        benchmark_path
    )

    pair_count = validate_benchmark_pairs(
        records=records,
        source_language=source_language,
        target_language=target_language,
        require_complete_pairs=benchmark.get(
            "require_complete_pairs",
            True
        )
    )

    validate_output_paths(
        experiment_config
    )

    return PreflightResult(
        experiment_name=experiment[
            "name"
        ],
        benchmark_record_count=len(
            records
        ),
        pair_count=pair_count,
        source_language=source_language,
        target_language=target_language,
        model_name=model_config.model_name,
        ready=True
    )


def parse_arguments() -> argparse.Namespace:
    """Command-line argumentlerini parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Run baseline experiment preflight checks."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/baseline.yaml"
        ),
        help="Baseline experiment config path.",
    )

    return parser.parse_args()


def main() -> None:
    """Preflight CLI entry point."""

    from src.experiments.run_baseline import (
        load_experiment_config,
    )

    args = parse_arguments()

    experiment_config = load_experiment_config(
        args.config
    )

    result = run_preflight(
        experiment_config
    )

    print(
        "Preflight passed."
    )

    print(
        f"Experiment: {result.experiment_name}"
    )

    print(
        f"Model: {result.model_name}"
    )

    print(
        f"Benchmark records: {result.benchmark_record_count}"
    )

    print(
        f"Pairs: {result.pair_count}"
    )

    print(
        f"Languages: {result.source_language} -> {result.target_language}"
    )


if __name__ == "__main__":
    main()
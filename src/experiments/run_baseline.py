"""Baseline experiment pipeline'ını tek komuttan çalıştırır.

Bu modül baseline experiment config dosyasını okuyarak:

1. Experiment yapılandırmasını yükler.
2. Benchmark test split'ini yükler.
3. Model config'i yükler.
4. Base model ve tokenizer'ı yükler.
5. Baseline inference çalıştırır.
6. Prediction artifact'ını kaydeder.
7. Capability ve reliability evaluation çalıştırır.
8. Evaluation artifact'larını kaydeder.
9. Experiment config snapshot ve run metadata üretir.

Amaç baseline experiment'ın manuel komut zinciri yerine tek, reproducible
entry point üzerinden çalıştırılmasını sağlamaktır.
"""



from __future__ import annotations

import argparse 
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data.build_benchmark import load_records
from src.evaluation.evaluate_predictions import (
    evaluate_predictions,
    save_evaluation_artifacts
)
from src.evaluation.run_inference import (
    run_inference,
    save_predictions
)
from src.models.load_config import (
    load_model_config,
    load_model_config_dict
)
from src.models.load_model import load_model_and_tokenizer
from src.experiments.preflight import run_preflight


def load_experiment_config(
    config_path: Path
) -> dict[str, Any]:
    """Experiment YAML dosyasını yükler ve Python dictionary olarak döndürür.

    Dosyanın root yapısı mapping olmalıdır.
    mapping → key:value yapısına sahip sözlük benzeri yapı demektir.
    Dosya yoksa FileNotFoundError, geçersiz yapı varsa ValueError oluşturur.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment config not found: {config_path}"
        )

    with config_path.open('r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

        if not isinstance(data, dict): 
            raise ValueError(
            "Experiment config must be a mapping."
        )

    return data




def set_reproducibility_seed(
    seed: int
) -> None:
    """Python ve PyTorch random seed değerlerini sabitler."""

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def save_config_snapshot(
    config: dict[str, Any],
    output_path: Path,
    overwrite: bool = False
) -> None:
    """Experiment config'in o run sırasında kullanılan kopyasını YAML olarak kaydeder.

    Amaç, experiment daha sonra tekrar incelendiğinde hangi config ayarlarıyla
    çalıştırıldığını aynen görebilmektir.

    `sort_keys=False`:
    YAML key'lerinin alfabetik olarak yeniden sıralanmasını engeller ve
    config içindeki mevcut sırayı korur.

    `allow_unicode=True`:
    Azərbaycan karakterleri gibi Unicode karakterlerin doğrudan ve okunabilir
    biçimde YAML dosyasına yazılmasını sağlar.

    `overwrite=False` iken dosya zaten varsa FileExistsError oluşturur.
    Gerekli parent klasörleri otomatik oluşturulur.
    """

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Config snapshot already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(
            config, 
            file,
            sort_keys=False,
            allow_unicode=True
        )



def build_run_metadata(
    experiment_config: dict[str, Any],
    model_config: dict[str, Any],
    prediction_count: int
) -> dict[str, Any]:
    """Baseline run için reproducibility metadata bilgilerini oluşturur.

    Bu metadata, experiment'ın hangi ayarlarla ve hangi modelle çalıştırıldığını
    daha sonra tekrar görebilmek için kullanılır.

    Kaydedilen bilgiler:
    - experiment adı
    - UTC timestamp
    - random seed
    - source ve target language
    - benchmark split ve input path
    - toplam prediction sayısı
    - kullanılan model config

    Örnek:
    {
        "experiment_name": "baseline_qwen",
        "timestamp_utc": "2026-08-22T11:20:00+00:00",
        "seed": 42,
        "source_language": "en",
        "target_language": "az",
        "benchmark_split": "test",
        "benchmark_input": "data/benchmark/test.jsonl",
        "prediction_count": 200,
        "model_config": {...}
    }
    """

    return {
        "experiment_name": experiment_config[
            "experiment"
        ][
            "name"
        ],
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "seed": experiment_config[
            "experiment"
        ][
            "seed"
        ],
        "source_language": experiment_config[
            "languages"
        ][
            "source"
        ],
        "target_language": experiment_config[
            "languages"
        ][
            "target"
        ],
        "benchmark_split": experiment_config[
            "benchmark"
        ].get(
            "split"
        ),
        "benchmark_input": experiment_config[
            "benchmark"
        ][
            "input_path"
        ],
        "prediction_count": prediction_count,
        "model_config": model_config
    }



def save_run_metadata(
    metadata: dict[str, Any],
    output_path: Path,
    overwrite: bool = False
) -> None:
    """Run metadata dictionary'sini JSON artifact olarak kaydeder."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Run metadata already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        'w',
        encoding='utf-8'
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

        file.write(
            '\n'
        )



def run_baseline_experiment(
    experiment_config: dict[str, Any]
) -> dict[str, Any]:
    """Baseline inference ve evaluation pipeline'ını baştan sona çalıştırır.

    `artifacts`, experiment sırasında üretilen ve daha sonra saklanan çıktı
    dosyalarını ifade eder.

    Örnek artifacts:
    - predictions.jsonl
    - evaluation sonuçları
    - experiment_config.yaml
    - run_metadata.json

    Akış sırası:
    1. Experiment config doğrulanır.
    2. Experiment, language, model, benchmark, inference, evaluation ve
       artifact ayarları ayrılır.
    3. Random seed sabitlenir.
    4. Benchmark input path, model config path ve output path'ler hazırlanır.
    5. Benchmark kayıtları yüklenir.
    6. Model config yüklenir.
    7. Model ve tokenizer yüklenir.
    8. Benchmark üzerinde inference çalıştırılır.
    9. Prediction kayıtları kaydedilir.
    10. Source ve target language bilgileri alınır.
    11. Capability ve reliability evaluation çalıştırılır.
    12. Evaluation artifact'ları kaydedilir.
    13. Config'te aktifse experiment config snapshot kaydedilir.
    14. Config'te aktifse run metadata oluşturulur ve kaydedilir.
    15. Prediction ve evaluation sonuçları dictionary olarak döndürülür.

    Bu fonksiyon baseline experiment'ın tek ana çalışma noktasıdır. 
    """

    run_preflight(
        experiment_config
    )

    experiment = experiment_config['experiment']    

    languages = experiment_config['languages']

    model_section = experiment_config['model']

    benchmark = experiment_config['benchmark']

    inference = experiment_config['inference']  

    evaluation_section = experiment_config['evaluation']

    artifacts = experiment_config['artifacts']  

    seed = experiment['seed']

    set_reproducibility_seed(seed)
    

    benchmark_path = Path(
        benchmark["input_path"]
    )

    model_config_path = Path(
        model_section["config_path"]
    )

    prediction_output_path = Path(
        inference["output_path"]
    )

    evaluation_output_dir = Path(
        evaluation_section["output_dir"]
    )

    benchmark_records = load_records(benchmark_path)

    if not benchmark_records:
        raise ValueError(
            "Benchmark input contains no records."
        )

    model_config = load_model_config(model_config_path)

    raw_model_config = load_model_config_dict(model_config_path)

    model, tokenizer = load_model_and_tokenizer(model_config)

    predictions = run_inference(
        records=benchmark_records,
        model=model,
        tokenizer=tokenizer,
        config=model_config
    )

    save_predictions(
        predictions=predictions,
        output_path=prediction_output_path,
        overwrite=inference.get(
            'overwrite',
            False
        )
    )

    source_language = languages[
        "source"
    ]

    target_language = languages[
        "target"
    ]

    evaluation = evaluate_predictions(
        predictions=predictions,
        source_language=source_language,
        target_language=target_language
    )

    save_evaluation_artifacts(
        evaluation=evaluation,
        output_dir=evaluation_output_dir,
        overwrite=evaluation_section.get(
            'overwrite',
            False
        )
    )

    if artifacts.get(
        'save_config_snapshot',
        False
    ):
        save_config_snapshot(
            config=experiment_config,
            output_path=(
                prediction_output_path.parent
                / "experiment_config.yaml"
            ),
            overwrite=inference.get(
                "overwrite",
                False
            )
        )

    if artifacts.get(
        "save_run_metadata",
        False
    ):
        metadata = build_run_metadata(
            experiment_config=experiment_config,
            model_config=raw_model_config,
            prediction_count=len(
                predictions
            )
        )

        save_run_metadata(
            metadata=metadata,
            output_path=(
                prediction_output_path.parent
                / "run_metadata.json"
            ),
            overwrite=inference.get(
                "overwrite",
                False
            )
        )

    return {
        "predictions": predictions,
        "evaluation": evaluation
    }


def parse_arguments() -> argparse.Namespace:
    """Command-line argumentlerini parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the baseline capability and reliability experiment."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/baseline.yaml"
        ),
        help="Baseline experiment config path."
    )

    return parser.parse_args()


def main() -> None:
    """Baseline experiment CLI entry point."""

    args = parse_arguments()

    experiment_config = load_experiment_config(
        args.config
    )

    result = run_baseline_experiment(
        experiment_config
    )

    prediction_count = len(
        result["predictions"]
    )

    print(
        f"Baseline experiment completed: {prediction_count} predictions."
    )


if __name__ == "__main__":
    main()
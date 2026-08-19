"""Benchmark kayıtları üzerinde model inference çalıştırır.

Bu modül final benchmark kayıtlarını yükler, her soru için model cevabı üretir
ve sonuçları reproducible evaluation artifact olarak JSONL formatında kaydeder.

Her prediction kaydı original benchmark kimliklerini, reference answer değerini
ve model tarafından üretilen cevabı birlikte saklar. Böylece sonraki capability,
reliability ve error analysis aşamaları aynı inference çıktısını kullanabilir.
"""


from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.benchmark_record import BenchmarkRecord
from src.data.build_benchmark import load_records
from src.models.generate import generate_response
from src.models.load_config import load_model_config
from src.models.load_model import load_model_and_tokenizer
from src.models.model_config import ModelConfig


@dataclass(frozen=True)
class PredictionRecord:
    """Tek bir benchmark item'ı için model prediction sonucunu temsil eder.

    Alanlar:
        item_id:
            Original benchmark item kimliği.

        pair_id:
            EN ve AZ semantic pair kimliği.

        language:
            Benchmark item'ının dili.

        task:
            Benchmark task ailesi.

        question:
            Modele verilen original benchmark sorusu.

        reference_answer:
            Benchmark içindeki beklenen doğru cevap.

        prediction:
            Model tarafından üretilen cevap.

        metadata:
            Benchmark kaydından korunan ek metadata alanları.
    """

    item_id: str
    pair_id: str
    language: str
    task: str
    question: str
    reference_answer: str
    prediction: str
    metadata: dict[str, Any]


    def to_dict(self) -> dict[str, Any]:
        """PredictionRecord nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)



def create_prediction_record(
    record: BenchmarkRecord,
    prediction: str
) -> PredictionRecord:
    """Benchmark kaydı ve model cevabından PredictionRecord oluşturur.

    Original benchmark alanlarını aynen korur ve modelin ürettiği cevabı
    `prediction` alanına ekler.

    Örnek:
    record:
        item_id = "reasoning_001_en"
        question = "What is 2 + 2?"
        reference_answer = "4"

    prediction:
        "4"

    Sonuç:
        PredictionRecord(
            item_id="reasoning_001_en",
            question="What is 2 + 2?",
            reference_answer="4",
            prediction="4",
            ...
        )

    Boş prediction değerine izin verilir; çünkü modelin hiçbir içerik
    üretmemesi de reliability analysis için anlamlı olabilir.
    """
    

    return PredictionRecord(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        question=record.question,
        reference_answer=record.reference_answer,
        prediction=prediction,
        metadata=dict(record.metadata),
    )


def run_inference(
    records: list[BenchmarkRecord],
    model: Any,
    tokenizer: Any,
    config: ModelConfig
) -> list[PredictionRecord]:
    """Benchmark kayıtları üzerinde sıralı model inference çalıştırır.

    Her BenchmarkRecord için aynı model, tokenizer ve generation config kullanılır.
    Model cevabı alınır ve ilgili benchmark kaydıyla birlikte PredictionRecord'a
    dönüştürülür.

    Örnek:
    records:
        reasoning_001_en → "What is 2 + 2?"
        reasoning_001_az → "2 + 2 neçə edir?"

    Model cevapları:
        "4"
        "4"

    Sonuç:
        [
            PredictionRecord(..., prediction="4"),
            PredictionRecord(..., prediction="4"),
        ]

    EN ve AZ kayıtlarında aynı inference protokolü kullanıldığı için karşılaştırma
    daha kontrollü olur. Kayıtların sırası prediction listesinde korunur.
    """

    predictions: list[PredictionRecord] = []

    for record in records:
        response = generate_response(
            prompt=record.question,
            model=model,
            tokenizer=tokenizer,
            config=config
        )

        predictions.append(
            create_prediction_record(
                record=record,
                prediction=response
            )
        )

    return predictions


def save_predictions(
    predictions: list[PredictionRecord],
    output_path: Path,
    overwrite: bool = False,
) -> None:
    """Prediction kayıtlarını JSONL dosyasına kaydeder.

    Parent klasör mevcut değilse oluşturulur.

    Output dosyası zaten mevcutsa ve overwrite=False ise mevcut evaluation
    artifact'ının yanlışlıkla üzerine yazılmasını önlemek için FileExistsError
    oluşturur.
    """

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Prediction output already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for prediction in predictions:
            file.write(
                json.dumps(
                    prediction.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_arguments() -> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştırır."""

    parser = argparse.ArgumentParser(
        description="Benchmark üzerinde model inference çalıştırır."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Inference uygulanacak benchmark JSONL dosyası."
    )

    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/base.yaml"),
        help="Model YAML config dosyasının yolu."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Prediction JSONL çıktısının yolu."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Mevcut prediction output dosyasının üzerine yazılmasına izin verir."
    )

    return parser.parse_args()


def main() -> None:
    """Benchmark inference sürecinin ana giriş noktasını çalıştırır."""

    args = parse_arguments()

    records = load_records(
        args.input
    )

    if not records:
        raise ValueError(
            "Inference input benchmark cannot be empty."
        )

    model_config = load_model_config(
        args.model_config
    )

    model, tokenizer = load_model_and_tokenizer(
        model_config
    )

    predictions = run_inference(
        records=records,
        model=model,
        tokenizer=tokenizer,
        config=model_config,
    )

    save_predictions(
        predictions=predictions,
        output_path=args.output,
        overwrite=args.overwrite,
    )

    print(
        f"Inference completed for {len(predictions)} records. "
        f"Predictions written to {args.output}"
    )


if __name__ == "__main__":
    main()
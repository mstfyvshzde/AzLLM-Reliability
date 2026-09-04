"""Prediction artifact'larını capability ve reliability açısından değerlendirir.

Bu modül daha önce üretilmiş PredictionRecord JSONL dosyasını okuyarak
evaluation pipeline'ını tek noktadan çalıştırır.

Pipeline:

    1. Prediction kayıtlarını yükler.
    2. Exact-match capability evaluation çalıştırır.
    3. Language, task, category ve difficulty özetleri üretir.
    4. EN-AZ paired capability analizini oluşturur.
    5. Abstention reliability evaluation çalıştırır.
    6. Reliability group metriclerini hesaplar.
    7. EN-AZ paired reliability analizini oluşturur.
    8. Item-level ve aggregate artifact'ları JSON/JSONL olarak kaydeder.

Bu dosya model inference çalıştırmaz.

Input:
    run_inference.py tarafından üretilmiş prediction JSONL artifact'ı.

Output:
    reproducible evaluation artifact'ları.
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.exact_match import (
    ExactMatchResult,
    evaluate_exact_match,
    summarize_exact_match
)
from src.evaluation.group_metrics import (
    calculate_language_gap,
    summarize_by_category,
    summarize_by_difficulty,
    summarize_by_language,
    summarize_by_task,
    summarize_language_task_matrix
)
from src.evaluation.paired_metrics import (
    PairedExactMatchResult,
    evaluate_paired_exact_match,
    summarize_paired_results
)
from src.evaluation.run_inference import PredictionRecord
from src.reliability.abstention_metrics import (
    AbstentionResult,
    evaluate_abstention,
    summarize_abstention_results
)
from src.reliability.group_metrics import (
    calculate_reliability_language_gap,
    summarize_reliability_by_category,
    summarize_reliability_by_difficulty,
    summarize_reliability_by_language,
    summarize_reliability_by_task
)
from src.reliability.paired_metrics import (
    PairedReliabilityResult,
    evaluate_paired_reliability,
    summarize_paired_reliability
)

from src.evaluation.short_answer_match import (
    ShortAnswerMatchResult,
    evaluate_short_answer_matches,
    filter_short_answer_records,
    summarize_short_answer_matches,
)
from src.evaluation.semantic_answer_match import (
    SemanticAnswerMatchResult,
    evaluate_semantic_answer_matches,
    filter_semantic_answer_records,
    summarize_semantic_answer_matches,
)
from src.evaluation.task_aware_capability import (
    TaskAwareCapabilityResult,
    evaluate_task_aware_capability,
    summarize_task_aware_capability,
)
from src.evaluation.semantic_adjudication import (
    load_semantic_adjudication_decisions,
)
from src.evaluation.paired_task_aware_capability import (
    PairedTaskAwareCapabilityResult,
    evaluate_paired_task_aware_capability,
    summarize_paired_task_aware_capability,
)
from src.evaluation.instruction_following_match import (
    InstructionFollowingResult,
    evaluate_instruction_following,
    summarize_instruction_following,
)
from src.analysis.failure_analysis import (
    FailureAnalysisResult,
    analyze_failures,
    summarize_failures,
)
from src.analysis.paired_failure_analysis import (
    PairedFailureAnalysisResult,
    evaluate_paired_failures,
    summarize_paired_failures,
)

def load_predictions(
    input_path: Path
) -> list[PredictionRecord]:
    """Prediction JSONL dosyasını PredictionRecord listesine yükler.

    Dosyayı satır satır okur, her JSON satırını PredictionRecord nesnesine
    dönüştürür ve tüm kayıtları `records` listesinde toplar.

    Örnek JSONL satırı:
    {
        "item_id": "reasoning_001_en",
        "pair_id": "reasoning_001",
        "language": "en",
        "task": "reasoning",
        "question": "What is 2 + 2?",
        "reference_answer": "4",
        "prediction": "4",
        "metadata": {}
    }

    Sonuç:
    [
        PredictionRecord(
            item_id="reasoning_001_en",
            pair_id="reasoning_001",
            language="en",
            task="reasoning",
            prediction="4",
            ...
        )
    ]

    Boş satırlar atlanır.
    Dosya bulunamazsa FileNotFoundError, bozuk JSON veya eksik alan varsa
    ValueError oluşturur.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {input_path}"
        )

    records: list[PredictionRecord] = []

    with input_path.open('r', encoding='utf-8') as file:
        for line_number,line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try: 
                data = json.loads(
                    stripped
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in '{input_path}'."
                ) from error


            try:
                record = PredictionRecord(
                    item_id=data["item_id"],
                    pair_id=data["pair_id"],
                    language=data["language"],
                    task=data["task"],
                    question=data["question"],
                    reference_answer=data["reference_answer"],
                    prediction=data["prediction"],
                    metadata=data.get(
                        "metadata",
                        {}
                    )
                )

            except KeyError as error:
                missing_field = error.args[0]

                raise ValueError(
                    f"Missing prediction field '{missing_field}' "
                    f"at line {line_number}."
                ) from error

            records.append(
                record
            )

    return records



def save_json(
    data: dict[str, Any],
    output_path: Path,
    overwrite: bool = False
) -> None:
    """Dictionary verisini JSON dosyası olarak kaydeder.

    Örnek:
    data = {
        "accuracy": 0.82,
        "total": 100
    }

    output_path = Path("outputs/summary.json")

    Sonuç:
    outputs/summary.json

    {
      "accuracy": 0.82,
      "total": 100
    }

    `overwrite=False` iken dosya zaten varsa üzerine yazmaz ve
    FileExistsError oluşturur.

    Gerekli parent klasörleri otomatik oluşturur.
    """

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(
            data,
            file, 
            ensure_ascii=False,
            indent=2
        )

        file.write('\n')



def save_jsonl(
    records: list[
        ExactMatchResult
        | FailureAnalysisResult
        | PairedFailureAnalysisResult
        | PairedTaskAwareCapabilityResult
        | TaskAwareCapabilityResult
        | InstructionFollowingResult
        | SemanticAnswerMatchResult
        | ShortAnswerMatchResult
        | PairedExactMatchResult
        | AbstentionResult
        | PairedReliabilityResult
    ],
    output_path: Path,
    overwrite: bool = False
) -> None:
    """Evaluation dataclass kayıtlarını JSONL dosyası olarak kaydeder.

    Her record ayrı bir JSON satırı olarak yazılır.

    Örnek:
    records = [
        ExactMatchResult(...),
        AbstentionResult(...)
    ]

    Çıktı:
    {"item_id": "...", "exact_match": 1, ...}
    {"item_id": "...", "outcome": "correct_answer", ...}

    `overwrite=False` iken dosya zaten varsa üzerine yazmaz ve
    FileExistsError oluşturur.

    Gerekli parent klasörleri otomatik oluşturur.
    """

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open('w', encoding='utf-8') as file:
        for record in records:
            json.dump(
                record.to_dict(),
                file,
                ensure_ascii=False
            )

            file.write('\n')


def build_capability_summary(
    results: list[ExactMatchResult],
    source_language: str = 'en',
    target_language: str = 'az'
) -> dict[str, Any]:
    """Exact-match sonuçlarından capability summary artifact'ı oluşturur.

    Bu fonksiyon item-level exact-match sonuçlarını farklı analiz seviyelerinde
    özetleyip tek bir dictionary içinde toplar.

    Örnek çıktı:
    {
        "overall": {
            "total": 100,
            "correct": 72,
            "incorrect": 28,
            "accuracy": 0.72
        },

        "by_language": {
            "en": {
                "total": 50,
                "correct": 40,
                "incorrect": 10,
                "accuracy": 0.80
            },
            "az": {
                "total": 50,
                "correct": 32,
                "incorrect": 18,
                "accuracy": 0.64
            }
        },

        "by_task": {
            "reasoning": {
                "total": 40,
                "correct": 30,
                "incorrect": 10,
                "accuracy": 0.75
            },
            "factual_qa": {
                "total": 60,
                "correct": 42,
                "incorrect": 18,
                "accuracy": 0.70
            }
        },

        "by_category": {
            "logical_reasoning": {
                "total": 20,
                "correct": 16,
                "incorrect": 4,
                "accuracy": 0.80
            },
            "local_knowledge": {
                "total": 20,
                "correct": 11,
                "incorrect": 9,
                "accuracy": 0.55
            }
        },

        "by_difficulty": {
            "easy": {
                "total": 30,
                "correct": 27,
                "incorrect": 3,
                "accuracy": 0.90
            },
            "medium": {
                "total": 40,
                "correct": 29,
                "incorrect": 11,
                "accuracy": 0.725
            },
            "hard": {
                "total": 30,
                "correct": 16,
                "incorrect": 14,
                "accuracy": 0.53
            }
        },

        "language_task_matrix": {
            "en": {
                "reasoning": {
                    "total": 20,
                    "correct": 17,
                    "incorrect": 3,
                    "accuracy": 0.85
                }
            },
            "az": {
                "reasoning": {
                    "total": 20,
                    "correct": 13,
                    "incorrect": 7,
                    "accuracy": 0.65
                }
            }
        },

        "language_gap": {
            "source_accuracy": 0.80,
            "target_accuracy": 0.64,
            "absolute_gap": 0.16
        }
    }

    Yani tek fonksiyonla modelin genel capability sonucunu, dil bazlı performansını,
    task/category/difficulty kırılımlarını ve EN-AZ language gap değerini üretir.

    Boş result listesi verilirse ValueError oluşturur.
    """

    if not results:
        raise ValueError(
            "Cannot build capability summary from empty results."
        )

    return {
        "overall": summarize_exact_match(
            results
        ),
        "by_language": summarize_by_language(
            results
        ),
        "by_task": summarize_by_task(
            results
        ),
        "by_category": summarize_by_category(
            results
        ),
        "by_difficulty": summarize_by_difficulty(
            results
        ),
        "language_task_matrix": summarize_language_task_matrix(
            results
        ),
        "language_gap": calculate_language_gap(
            results,
            source_language=source_language,
            target_language=target_language
        )
    }


def build_reliability_summary(
    results: list[AbstentionResult],
    source_language: str = 'en',
    target_language: str = 'az'
) -> dict[str, Any]:
    """Abstention sonuçlarından reliability summary artifact'ı oluşturur.

    Bu fonksiyon item-level AbstentionResult kayıtlarını farklı analiz
    seviyelerinde özetleyip tek bir dictionary içinde toplar.

    Örnek çıktı:
    {
        "overall": {
            "total": 100,
            "correct_answer": 50,
            "correct_abstention": 20,
            "under_answering": 10,
            "over_answering": 15,
            "empty_response": 5,
            "abstention_accuracy": 0.70,
            "over_answering_rate": 0.375,
            "under_answering_rate": 0.167
        },

        "by_language": {
            "en": {
                "total": 50,
                "abstention_accuracy": 0.78,
                "over_answering_rate": 0.25,
                "under_answering_rate": 0.12,
                "empty_response_rate": 0.02
            },
            "az": {
                "total": 50,
                "abstention_accuracy": 0.62,
                "over_answering_rate": 0.50,
                "under_answering_rate": 0.22,
                "empty_response_rate": 0.08
            }
        },

        "by_task": {
            "reasoning": {
                "abstention_accuracy": 0.75,
                "over_answering_rate": 0.30,
                "under_answering_rate": 0.15
            },
            "factual_qa": {
                "abstention_accuracy": 0.65,
                "over_answering_rate": 0.45,
                "under_answering_rate": 0.20
            }
        },

        "by_category": {
            "logical_reasoning": {
                "abstention_accuracy": 0.80
            },
            "local_knowledge": {
                "abstention_accuracy": 0.55
            }
        },

        "by_difficulty": {
            "easy": {
                "abstention_accuracy": 0.85
            },
            "medium": {
                "abstention_accuracy": 0.70
            },
            "hard": {
                "abstention_accuracy": 0.50
            }
        },

        "language_gap": {
            "source_abstention_accuracy": 0.78,
            "target_abstention_accuracy": 0.62,
            "abstention_accuracy_gap": 0.16,

            "source_over_answering_rate": 0.25,
            "target_over_answering_rate": 0.50,
            "over_answering_rate_gap": -0.25,

            "source_under_answering_rate": 0.12,
            "target_under_answering_rate": 0.22,
            "under_answering_rate_gap": -0.10
        }
    }

    Yani tek fonksiyonla modelin genel reliability davranışını, dil/task/category/
    difficulty kırılımlarını ve EN-AZ reliability gap değerlerini üretir.

    Boş result listesi verilirse ValueError oluşturur.
    """

    if not results:
        raise ValueError(
            "Cannot build reliability summary from empty results."
        )

    return {
        "overall": summarize_abstention_results(
            results
        ),
        "by_language": summarize_reliability_by_language(
            results
        ),
        "by_task": summarize_reliability_by_task(
            results
        ),
        "by_category": summarize_reliability_by_category(
            results
        ),
        "by_difficulty": summarize_reliability_by_difficulty(
            results
        ),
        "language_gap": calculate_reliability_language_gap(
            results,
            source_language=source_language,
            target_language=target_language
        )
    }



def evaluate_predictions(
    predictions: list[PredictionRecord],
    source_language: str = 'en',
    target_language: str = 'az',
    semantic_adjudication_decisions: dict[str, int] | None = None,
    binary_adjudication_decisions: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Prediction kayıtları üzerinde tüm capability ve reliability evaluation'ı çalıştırır.

    Her prediction için exact-match ve abstention analizlerini yapar,
    ardından EN-AZ pair karşılaştırmalarını ve aggregate summary'leri oluşturur.

    Örnek:
    predictions:
        reasoning_001_en → prediction="4"
        reasoning_001_az → prediction="4"

    Sonuç:
        {
            "exact_match_results": [...],
            "paired_capability_results": [...],
            "abstention_results": [...],
            "paired_reliability_results": [...],
            "capability_summary": {...},
            "paired_capability_summary": {...},
            "reliability_summary": {...},
            "paired_reliability_summary": {...}
        }

    Boş prediction listesi verilirse ValueError oluşturur.
    """

    if not predictions:
        raise ValueError(
            "Cannot evaluate empty prediction records."
        )

    exact_match_results = evaluate_exact_match(
        predictions
    )

    short_answer_records = filter_short_answer_records(
        predictions
    )   

    short_answer_results = evaluate_short_answer_matches(
        short_answer_records
    )

    short_answer_summary = summarize_short_answer_matches(
        short_answer_results
    )

    paired_capability_results = evaluate_paired_exact_match(
        exact_match_results,
        source_language=source_language,
        target_language=target_language
    )

    abstention_results = evaluate_abstention(
        predictions
    )

    paired_reliability_results = evaluate_paired_reliability(
        abstention_results,
        source_language=source_language,
        target_language=target_language
    )

    capability_summary = build_capability_summary(
        exact_match_results,
        source_language=source_language,
        target_language=target_language
    )

    reliability_summary = build_reliability_summary(
        abstention_results,
        source_language=source_language,
        target_language=target_language
    )

    paired_capability_summary = summarize_paired_results(
        paired_capability_results
    )

    paired_reliability_summary = summarize_paired_reliability(
        paired_reliability_results
    )

    semantic_answer_records = filter_semantic_answer_records(
        predictions
    )

    semantic_answer_results = evaluate_semantic_answer_matches(
        semantic_answer_records
    )

    task_aware_results = evaluate_task_aware_capability(
        predictions,
        semantic_adjudication_decisions=(
            semantic_adjudication_decisions
        ),
        binary_adjudication_decisions=(
            binary_adjudication_decisions
        ),
    )

    paired_task_aware_results = evaluate_paired_task_aware_capability(
        task_aware_results,
        source_language=source_language,
        target_language=target_language,
    )

    paired_task_aware_summary = summarize_paired_task_aware_capability(
        paired_task_aware_results
    )

    task_aware_summary = summarize_task_aware_capability(
        task_aware_results
    )

    instruction_following_records = [
        record
        for record in predictions
        if record.task == "instruction_following"
    ]

    instruction_following_results = evaluate_instruction_following(
        instruction_following_records
    )

    instruction_following_summary = (
        summarize_instruction_following(
            instruction_following_results
        )
        if instruction_following_results
        else {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "accuracy": None,
        }
    )
    

    semantic_answer_summary = (
        summarize_semantic_answer_matches(
            semantic_answer_results
        )
        if semantic_answer_results
        else {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "accuracy": None,
        }
    )

    capability_scores = {
        result.item_id: result.correct
        for result in task_aware_results
    }

    reliability_outcomes = {
        result.item_id: result.outcome
        for result in abstention_results
    }

    failure_analysis_results = analyze_failures(
        predictions,
        capability_scores=capability_scores,
        reliability_outcomes=reliability_outcomes,
    )

    failure_analysis_summary = summarize_failures(
        failure_analysis_results
    )

    paired_failure_analysis_results = (
        evaluate_paired_failures(
            failure_analysis_results,
            source_language=source_language,
            target_language=target_language,
        )
    )

    paired_failure_analysis_summary = (
        summarize_paired_failures(
            paired_failure_analysis_results
        )
    )

    return {
        "exact_match_results": exact_match_results,
        "paired_capability_results": paired_capability_results,
        "abstention_results": abstention_results,
        "paired_reliability_results": paired_reliability_results,
        "capability_summary": capability_summary,
        "paired_capability_summary": paired_capability_summary,
        "reliability_summary": reliability_summary,
        "paired_reliability_summary": paired_reliability_summary,
        "short_answer_results": short_answer_results,
        "short_answer_summary": short_answer_summary,
        "semantic_answer_results": semantic_answer_results,
        "semantic_answer_summary": semantic_answer_summary,
        "task_aware_results": task_aware_results,
        "task_aware_summary": task_aware_summary,
        "paired_task_aware_results": paired_task_aware_results,
        "paired_task_aware_summary": paired_task_aware_summary,
        "primary_capability_summary": task_aware_summary,
        "primary_paired_capability_summary": paired_task_aware_summary,
        "instruction_following_results": instruction_following_results,
        "instruction_following_summary": instruction_following_summary,
        "failure_analysis_results": failure_analysis_results,
        "failure_analysis_summary": failure_analysis_summary,
        "paired_failure_analysis_results": (
            paired_failure_analysis_results
        ),
        "paired_failure_analysis_summary": (
            paired_failure_analysis_summary
        )
    }



def save_evaluation_artifacts(
    evaluation: dict[str, Any],
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """Evaluation pipeline çıktılarının tamamını artifact dosyalarına kaydeder."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_jsonl(
        evaluation[
            "exact_match_results"
        ],
        output_dir
        / "exact_match_results.jsonl",
        overwrite=overwrite
    )

    save_jsonl(
        evaluation[
            "paired_capability_results"
        ],
        output_dir
        / "paired_capability_results.jsonl",
        overwrite=overwrite
    )

    save_jsonl(
        evaluation[
            "abstention_results"
        ],
        output_dir
        / "abstention_results.jsonl",
        overwrite=overwrite
    )

    save_jsonl(
        evaluation[
            "paired_reliability_results"
        ],
        output_dir
        / "paired_reliability_results.jsonl",
        overwrite=overwrite
    )

    save_json(
        evaluation[
            "capability_summary"
        ],
        output_dir
        / "capability_summary.json",
        overwrite=overwrite
    )

    save_json(
        evaluation[
            "paired_capability_summary"
        ],
        output_dir
        / "paired_capability_summary.json",
        overwrite=overwrite
    )

    save_json(
        evaluation[
            "reliability_summary"
        ],
        output_dir
        / "reliability_summary.json",
        overwrite=overwrite
    )

    save_json(
        evaluation[
            "paired_reliability_summary"
        ],
        output_dir
        / "paired_reliability_summary.json",
        overwrite=overwrite
    )

    save_jsonl(
        evaluation[
            "short_answer_results"
        ],
        output_dir
        / "short_answer_results.jsonl",
        overwrite=overwrite
    )

    save_json(
        evaluation[
            "short_answer_summary"
        ],
        output_dir
        / "short_answer_summary.json",
        overwrite=overwrite   
    )

    save_jsonl(
        evaluation[
            "semantic_answer_results"
        ],
        output_dir
        / "semantic_answer_results.jsonl",
        overwrite=overwrite
    )

    save_json(
        evaluation[
            "semantic_answer_summary"
        ],
        output_dir
        / "semantic_answer_summary.json",
        overwrite=overwrite
    )

    save_jsonl(
        evaluation[
            "task_aware_results"
        ],
        output_dir
        / "task_aware_results.jsonl",
        overwrite=overwrite
    )

    save_json(
        evaluation[
            "task_aware_summary"
        ],
        output_dir
        / "task_aware_summary.json",
        overwrite=overwrite
    )

    save_jsonl(
        evaluation[
            "paired_task_aware_results"
        ],
        output_dir
        / "paired_task_aware_results.jsonl",
        overwrite=overwrite
    )

    save_json(
        evaluation[
            "paired_task_aware_summary"
        ],
        output_dir
        / "paired_task_aware_summary.json",
        overwrite=overwrite,
    )

    save_json(
        evaluation[
            "primary_capability_summary"
        ],
        output_dir
        / "primary_capability_summary.json",
        overwrite=overwrite,
    )

    save_json(
        evaluation[
            "primary_paired_capability_summary"
        ],
        output_dir
        / "primary_paired_capability_summary.json",
        overwrite=overwrite,
    )

    save_jsonl(
        evaluation[
            "instruction_following_results"
        ],
        output_dir
        / "instruction_following_results.jsonl",
        overwrite=overwrite
    )
    save_json(
        evaluation[
            "instruction_following_summary"
        ],
        output_dir
        / "instruction_following_summary.json",
        overwrite=overwrite
    )

    save_jsonl(
        evaluation["failure_analysis_results"],
        output_dir / "failure_analysis_results.jsonl",
        overwrite=overwrite,
    )

    save_json(
        evaluation["failure_analysis_summary"],
        output_dir / "failure_analysis_summary.json",
        overwrite=overwrite,
    )

    save_jsonl(
        evaluation["paired_failure_analysis_results"],
        output_dir / "paired_failure_analysis_results.jsonl",
        overwrite=overwrite,
    )

    save_json(
        evaluation["paired_failure_analysis_summary"],
        output_dir / "paired_failure_analysis_summary.json",
        overwrite=overwrite,
    )


def parse_arguments() -> argparse.Namespace:
    """Command-line argumentlerini parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate model prediction artifacts for "
            "capability and reliability."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Prediction JSONL file."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for evaluation artifacts."
    )

    parser.add_argument(
        "--source-language",
        type=str,
        default="en",
        help="Source language code."
    )

    parser.add_argument(
        "--target-language",
        type=str,
        default="az",
        help="Target language code."
    )

    parser.add_argument(
        "--semantic-adjudication-decisions",
        type=Path,
        default=None,
        help=(
            "Optional semantic adjudication JSONL artifact "
            "for open semantic linguistic categories."
        ),
    )
    parser.add_argument(
        "--binary-adjudication-decisions",
        type=Path,
        default=None,
        help=(
            "Optional binary adjudication JSONL artifact "
            "for indirect Yes/No answers."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation artifacts."
    )

    return parser.parse_args()


def main() -> None:
    """Evaluation pipeline CLI entry point."""

    args = parse_arguments()

    predictions = load_predictions(
        args.input
    )

    if not predictions:
        raise ValueError(
            "Prediction file contains no records."
        )

    semantic_adjudication_decisions = None
    binary_adjudication_decisions = None

    if args.semantic_adjudication_decisions is not None:
        semantic_adjudication_decisions = (
            load_semantic_adjudication_decisions(
                str(args.semantic_adjudication_decisions)
            )
        )

    if args.binary_adjudication_decisions is not None:
        binary_adjudication_decisions = (
            load_semantic_adjudication_decisions(
                str(args.binary_adjudication_decisions)
            )
        )

    evaluation = evaluate_predictions(
        predictions=predictions,
        source_language=args.source_language,
        target_language=args.target_language,
        semantic_adjudication_decisions=(
            semantic_adjudication_decisions
        ),
        binary_adjudication_decisions=(
            binary_adjudication_decisions
        ),
    )

    save_evaluation_artifacts(
        evaluation=evaluation,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print(
        f"Evaluated {len(predictions)} predictions."
    )

    print(
        f"Artifacts saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()

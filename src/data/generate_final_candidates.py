"""Final benchmark candidate kayıtlarını generation planından oluşturur.

Bu modül iki girdiyi birleştirir:

    1. benchmark_generation_plan.jsonl
        -> pair_id
        -> task
        -> category
        -> difficulty
        -> answerability
        -> language bilgileri

    2. final_candidate_content.jsonl
        -> English question/reference answer
        -> Azerbaijani question/reference answer

Sonuç olarak her semantic pair için iki BenchmarkRecord oluşturulur:

    pair_id_en
    pair_id_az

Tüm yeni candidate kayıtları:

    metadata.review_status = "pending"

olarak başlar.

Bu aşama benchmark içeriğini otomatik approve etmez.
Human review daha sonra ayrı review pipeline'ında yapılır.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.benchmark_record import BenchmarkRecord
from src.data.generate_benchmark_plan import BenchmarkPlanRecord


DEFAULT_PLAN_PATH = Path(
    "data/processed/benchmark_generation_plan.jsonl"
)

DEFAULT_CONTENT_PATH = Path(
    "data/interim/final_candidate_content.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/candidates/final_candidates.jsonl"
)


@dataclass(frozen=True)
class CandidateContentRecord:
    """Tek bir semantic pair için EN-AZ candidate içeriğini temsil eder."""

    pair_id: str
    source_question: str
    source_reference_answer: str
    target_question: str
    target_reference_answer: str


def load_generation_plan(
    input_path: Path = DEFAULT_PLAN_PATH,
) -> list[BenchmarkPlanRecord]:
    """Generation plan JSONL dosyasını yükler."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Generation plan not found: {input_path}"
        )

    records: list[BenchmarkPlanRecord] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                data = json.loads(stripped)

            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid generation-plan JSON "
                    f"at line {line_number}."
                ) from error

            try:
                record = BenchmarkPlanRecord(
                    pair_id=data["pair_id"],
                    task=data["task"],
                    category=data["category"],
                    difficulty=data["difficulty"],
                    source_language=data["source_language"],
                    target_language=data["target_language"],
                    is_answerable=data["is_answerable"],
                )

            except KeyError as error:
                missing_field = error.args[0]

                raise ValueError(
                    "Missing generation-plan field "
                    f"'{missing_field}' "
                    f"at line {line_number}."
                ) from error

            records.append(record)

    return records


def _require_non_empty_string(
    value: Any,
    *,
    field_name: str,
    line_number: int,
) -> str:
    """Candidate content alanının boş olmayan string olduğunu doğrular."""

    if not isinstance(value, str):
        raise ValueError(
            f"Candidate field '{field_name}' "
            f"must be a string at line {line_number}."
        )

    stripped = value.strip()

    if not stripped:
        raise ValueError(
            f"Candidate field '{field_name}' "
            f"cannot be empty at line {line_number}."
        )

    return stripped


def load_candidate_content(
    input_path: Path = DEFAULT_CONTENT_PATH,
) -> list[CandidateContentRecord]:
    """EN-AZ candidate content JSONL dosyasını yükler."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Candidate content file not found: {input_path}"
        )

    records: list[CandidateContentRecord] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                data = json.loads(stripped)

            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid candidate-content JSON "
                    f"at line {line_number}."
                ) from error

            required_fields = (
                "pair_id",
                "source_question",
                "source_reference_answer",
                "target_question",
                "target_reference_answer",
            )

            for field_name in required_fields:
                if field_name not in data:
                    raise ValueError(
                        "Missing candidate-content field "
                        f"'{field_name}' "
                        f"at line {line_number}."
                    )

            records.append(
                CandidateContentRecord(
                    pair_id=_require_non_empty_string(
                        data["pair_id"],
                        field_name="pair_id",
                        line_number=line_number,
                    ),
                    source_question=_require_non_empty_string(
                        data["source_question"],
                        field_name="source_question",
                        line_number=line_number,
                    ),
                    source_reference_answer=_require_non_empty_string(
                        data["source_reference_answer"],
                        field_name="source_reference_answer",
                        line_number=line_number,
                    ),
                    target_question=_require_non_empty_string(
                        data["target_question"],
                        field_name="target_question",
                        line_number=line_number,
                    ),
                    target_reference_answer=_require_non_empty_string(
                        data["target_reference_answer"],
                        field_name="target_reference_answer",
                        line_number=line_number,
                    ),
                )
            )

    return records


def validate_unique_plan_pair_ids(
    plan: list[BenchmarkPlanRecord],
) -> None:
    """Generation plan içinde duplicate pair_id olmadığını doğrular."""

    seen: set[str] = set()

    for record in plan:
        if record.pair_id in seen:
            raise ValueError(
                "Duplicate pair_id in generation plan: "
                f"{record.pair_id}"
            )

        seen.add(record.pair_id)


def validate_unique_content_pair_ids(
    content: list[CandidateContentRecord],
) -> None:
    """Candidate content içinde duplicate pair_id olmadığını doğrular."""

    seen: set[str] = set()

    for record in content:
        if record.pair_id in seen:
            raise ValueError(
                "Duplicate pair_id in candidate content: "
                f"{record.pair_id}"
            )

        seen.add(record.pair_id)


def validate_plan_content_alignment(
    plan: list[BenchmarkPlanRecord],
    content: list[CandidateContentRecord],
) -> None:
    """Generation plan ile candidate content pair_id setlerini karşılaştırır."""

    validate_unique_plan_pair_ids(plan)
    validate_unique_content_pair_ids(content)

    plan_ids = {
        record.pair_id
        for record in plan
    }

    content_ids = {
        record.pair_id
        for record in content
    }

    missing_content = plan_ids - content_ids
    unexpected_content = content_ids - plan_ids

    if missing_content:
        preview = sorted(missing_content)[:10]

        raise ValueError(
            "Candidate content is missing planned pairs: "
            f"{preview}"
        )

    if unexpected_content:
        preview = sorted(unexpected_content)[:10]

        raise ValueError(
            "Candidate content contains unexpected pairs: "
            f"{preview}"
        )


def build_candidate_pair(
    plan_record: BenchmarkPlanRecord,
    content_record: CandidateContentRecord,
) -> tuple[BenchmarkRecord, BenchmarkRecord]:
    """Tek bir plan/content pair'inden EN ve AZ BenchmarkRecord üretir."""

    if plan_record.pair_id != content_record.pair_id:
        raise ValueError(
            "Plan/content pair_id mismatch: "
            f"{plan_record.pair_id!r} vs "
            f"{content_record.pair_id!r}"
        )

    shared_metadata = {
        "category": plan_record.category,
        "difficulty": plan_record.difficulty,
        "review_status": "pending",
        "is_answerable": plan_record.is_answerable,
    }

    source_item_id = (
        f"{plan_record.pair_id}_"
        f"{plan_record.source_language}"
    )

    target_item_id = (
        f"{plan_record.pair_id}_"
        f"{plan_record.target_language}"
    )

    source_record = BenchmarkRecord(
        item_id=source_item_id,
        pair_id=plan_record.pair_id,
        language=plan_record.source_language,
        task=plan_record.task,
        question=content_record.source_question,
        reference_answer=content_record.source_reference_answer,
        metadata=dict(shared_metadata),
    )

    target_record = BenchmarkRecord(
        item_id=target_item_id,
        pair_id=plan_record.pair_id,
        language=plan_record.target_language,
        task=plan_record.task,
        question=content_record.target_question,
        reference_answer=content_record.target_reference_answer,
        metadata=dict(shared_metadata),
    )

    return (
        source_record,
        target_record,
    )


def generate_final_candidates(
    plan: list[BenchmarkPlanRecord],
    content: list[CandidateContentRecord],
) -> list[BenchmarkRecord]:
    """Tüm generation planından final candidate BenchmarkRecord'ları üretir."""

    validate_plan_content_alignment(
        plan,
        content,
    )

    content_by_pair = {
        record.pair_id: record
        for record in content
    }

    candidates: list[BenchmarkRecord] = []

    for plan_record in plan:
        content_record = content_by_pair[
            plan_record.pair_id
        ]

        source_record, target_record = build_candidate_pair(
            plan_record,
            content_record,
        )

        candidates.extend(
            [
                source_record,
                target_record,
            ]
        )

    expected_records = len(plan) * 2

    if len(candidates) != expected_records:
        raise ValueError(
            "Generated candidate record count mismatch: "
            f"expected={expected_records}, "
            f"actual={len(candidates)}"
        )

    return candidates


def validate_generated_candidate_ids(
    records: list[BenchmarkRecord],
) -> None:
    """Generated candidate item_id değerlerinin unique olduğunu doğrular."""

    seen: set[str] = set()

    for record in records:
        if record.item_id in seen:
            raise ValueError(
                "Duplicate generated item_id: "
                f"{record.item_id}"
            )

        seen.add(record.item_id)


def save_final_candidates(
    records: list[BenchmarkRecord],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    overwrite: bool = False,
) -> None:
    """Final candidate BenchmarkRecord kayıtlarını JSONL olarak kaydeder."""

    validate_generated_candidate_ids(records)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            json.dump(
                record.to_dict(),
                file,
                ensure_ascii=False,
            )

            file.write("\n")


def summarize_final_candidates(
    records: list[BenchmarkRecord],
) -> dict[str, Any]:
    """Generated candidate kayıtlarının temel dağılımını özetler."""

    pair_ids = {
        record.pair_id
        for record in records
    }

    summary: dict[str, Any] = {
        "total_records": len(records),
        "total_pairs": len(pair_ids),
        "by_language": {},
        "by_task": {},
        "by_review_status": {},
    }

    for record in records:
        summary["by_language"][
            record.language
        ] = (
            summary["by_language"].get(
                record.language,
                0,
            )
            + 1
        )

        summary["by_task"][
            record.task
        ] = (
            summary["by_task"].get(
                record.task,
                0,
            )
            + 1
        )

        review_status = record.metadata.get(
            "review_status"
        )

        summary["by_review_status"][
            review_status
        ] = (
            summary["by_review_status"].get(
                review_status,
                0,
            )
            + 1
        )

    return summary


def parse_args() -> argparse.Namespace:
    """CLI argumentlarını parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate final EN-AZ benchmark candidates "
            "from the frozen generation plan."
        )
    )

    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Generation plan JSONL path.",
    )

    parser.add_argument(
        "--content",
        type=Path,
        default=DEFAULT_CONTENT_PATH,
        help="Candidate content JSONL path.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Generated candidate output JSONL path.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )

    return parser.parse_args()


def main() -> None:
    """CLI üzerinden final candidate generation çalıştırır."""

    args = parse_args()

    plan = load_generation_plan(
        args.plan
    )

    content = load_candidate_content(
        args.content
    )

    candidates = generate_final_candidates(
        plan,
        content,
    )

    save_final_candidates(
        candidates,
        args.output,
        overwrite=args.overwrite,
    )

    summary = summarize_final_candidates(
        candidates
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
"""Model-generated benchmark candidate content'ini toplar ve doğrular.

Bu modül candidate-generation request manifesti ile dışarıdan üretilmiş
model response kayıtlarını birleştirir.

Amaç:

    candidate_generation_requests.jsonl
        +
    raw_candidate_content.jsonl
        ↓
    validation
        ↓
    final_candidate_content.jsonl

Bu modül doğrudan bir LLM API çağrısı yapmaz.

Bunun nedeni generation provider'ını benchmark veri modelinden ayırmaktır.
İçerik OpenAI, Hugging Face, local model veya kontrollü human generation
ile üretilebilir; ancak final benchmark pipeline aynı kalır.

Beklenen raw content formatı:

    {
        "pair_id": "reasoning_0001",
        "source_question": "...",
        "source_reference_answer": "...",
        "target_question": "...",
        "target_reference_answer": "..."
    }

Bu aşamadaki kayıtlar henüz approved benchmark item değildir.
Human review daha sonraki aşamada yapılır.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.prepare_candidate_content import (
    CandidateGenerationRequest,
)


DEFAULT_REQUEST_PATH = Path(
    "data/interim/candidate_generation_requests.jsonl"
)

DEFAULT_RAW_CONTENT_PATH = Path(
    "data/interim/raw_candidate_content.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/interim/final_candidate_content.jsonl"
)


@dataclass(frozen=True)
class GeneratedCandidateContent:
    """Tek bir EN-AZ semantic pair için generated content kaydı."""

    pair_id: str
    source_question: str
    source_reference_answer: str
    target_question: str
    target_reference_answer: str

    def to_dict(self) -> dict[str, Any]:
        """Kaydı JSON-serializable dictionary biçimine dönüştürür."""

        return asdict(self)


def _require_non_empty_string(
    value: Any,
    *,
    field_name: str,
    line_number: int,
) -> str:
    """Alan değerinin boş olmayan string olduğunu doğrular."""

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            f"Field '{field_name}' must be a string "
            f"at line {line_number}."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"Field '{field_name}' cannot be empty "
            f"at line {line_number}."
        )

    return value


def load_generation_requests(
    input_path: Path = DEFAULT_REQUEST_PATH,
) -> list[CandidateGenerationRequest]:
    """Generation request JSONL artifact'ını yükler."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Generation request file not found: {input_path}"
        )

    records: list[
        CandidateGenerationRequest
    ] = []

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
                data = json.loads(
                    stripped
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid generation-request JSON "
                    f"at line {line_number}."
                ) from error

            required_fields = (
                "pair_id",
                "task",
                "category",
                "difficulty",
                "source_language",
                "target_language",
                "is_answerable",
                "instruction",
            )

            for field_name in required_fields:
                if field_name not in data:
                    raise ValueError(
                        "Missing generation-request field "
                        f"'{field_name}' "
                        f"at line {line_number}."
                    )

            records.append(
                CandidateGenerationRequest(
                    pair_id=_require_non_empty_string(
                        data["pair_id"],
                        field_name="pair_id",
                        line_number=line_number,
                    ),
                    task=_require_non_empty_string(
                        data["task"],
                        field_name="task",
                        line_number=line_number,
                    ),
                    category=_require_non_empty_string(
                        data["category"],
                        field_name="category",
                        line_number=line_number,
                    ),
                    difficulty=_require_non_empty_string(
                        data["difficulty"],
                        field_name="difficulty",
                        line_number=line_number,
                    ),
                    source_language=_require_non_empty_string(
                        data["source_language"],
                        field_name="source_language",
                        line_number=line_number,
                    ),
                    target_language=_require_non_empty_string(
                        data["target_language"],
                        field_name="target_language",
                        line_number=line_number,
                    ),
                    is_answerable=data[
                        "is_answerable"
                    ],
                    instruction=_require_non_empty_string(
                        data["instruction"],
                        field_name="instruction",
                        line_number=line_number,
                    ),
                )
            )

    return records


def load_raw_candidate_content(
    input_path: Path = DEFAULT_RAW_CONTENT_PATH,
) -> list[GeneratedCandidateContent]:
    """Model veya human generation çıktısı olan raw JSONL içeriğini yükler."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw candidate content file not found: {input_path}"
        )

    records: list[
        GeneratedCandidateContent
    ] = []

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
                data = json.loads(
                    stripped
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid raw candidate JSON "
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
                        "Missing raw candidate field "
                        f"'{field_name}' "
                        f"at line {line_number}."
                    )

            records.append(
                GeneratedCandidateContent(
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


def validate_unique_request_pair_ids(
    requests: list[CandidateGenerationRequest],
) -> None:
    """Request manifestinde duplicate pair_id olmadığını doğrular."""

    seen: set[str] = set()

    for record in requests:
        if record.pair_id in seen:
            raise ValueError(
                "Duplicate generation-request pair_id: "
                f"{record.pair_id}"
            )

        seen.add(
            record.pair_id
        )


def validate_unique_generated_pair_ids(
    records: list[GeneratedCandidateContent],
) -> None:
    """Generated content içinde duplicate pair_id olmadığını doğrular."""

    seen: set[str] = set()

    for record in records:
        if record.pair_id in seen:
            raise ValueError(
                "Duplicate generated-content pair_id: "
                f"{record.pair_id}"
            )

        seen.add(
            record.pair_id
        )


def validate_request_content_alignment(
    requests: list[CandidateGenerationRequest],
    records: list[GeneratedCandidateContent],
) -> None:
    """Request manifesti ile generated content pair setlerini doğrular."""

    validate_unique_request_pair_ids(
        requests
    )

    validate_unique_generated_pair_ids(
        records
    )

    request_ids = {
        record.pair_id
        for record in requests
    }

    generated_ids = {
        record.pair_id
        for record in records
    }

    missing = (
        request_ids
        - generated_ids
    )

    unexpected = (
        generated_ids
        - request_ids
    )

    if missing:
        raise ValueError(
            "Generated content is missing requested pairs: "
            f"{sorted(missing)[:10]}"
        )

    if unexpected:
        raise ValueError(
            "Generated content contains unexpected pairs: "
            f"{sorted(unexpected)[:10]}"
        )


def validate_language_distinction(
    records: list[GeneratedCandidateContent],
) -> None:
    """EN ve AZ question alanlarının yanlışlıkla birebir aynı olmadığını kontrol eder.

    Bu yalnızca basit bir structural guard'dır.
    Semantic-equivalence veya translation-quality değerlendirmesi değildir.
    """

    for record in records:
        if (
            record.source_question.strip()
            == record.target_question.strip()
        ):
            raise ValueError(
                "Source and target questions are identical "
                f"for pair: {record.pair_id}"
            )


def validate_generated_content(
    requests: list[CandidateGenerationRequest],
    records: list[GeneratedCandidateContent],
) -> None:
    """Generated candidate content için structural validation çalıştırır."""

    validate_request_content_alignment(
        requests,
        records,
    )

    validate_language_distinction(
        records
    )


def prepare_final_candidate_content(
    requests: list[CandidateGenerationRequest],
    records: list[GeneratedCandidateContent],
) -> list[GeneratedCandidateContent]:
    """Validated generated content'i final candidate-content listesine hazırlar."""

    validate_generated_content(
        requests,
        records,
    )

    content_by_pair = {
        record.pair_id: record
        for record in records
    }

    ordered_records = [
        content_by_pair[
            request.pair_id
        ]
        for request in requests
    ]

    return ordered_records


def save_candidate_content(
    records: list[GeneratedCandidateContent],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    overwrite: bool = False,
) -> None:
    """Validated candidate content'i JSONL olarak kaydeder."""

    if (
        output_path.exists()
        and not overwrite
    ):
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

            file.write(
                "\n"
            )


def summarize_candidate_content(
    requests: list[CandidateGenerationRequest],
    records: list[GeneratedCandidateContent],
) -> dict[str, Any]:
    """Generated content coverage bilgisini özetler."""

    request_by_pair = {
        request.pair_id: request
        for request in requests
    }

    summary: dict[str, Any] = {
        "total_requests": len(requests),
        "total_generated": len(records),
        "coverage": (
            len(records) / len(requests)
            if requests
            else None
        ),
        "by_task": {},
        "by_difficulty": {},
    }

    for record in records:
        request = request_by_pair[
            record.pair_id
        ]

        summary["by_task"][
            request.task
        ] = (
            summary["by_task"].get(
                request.task,
                0,
            )
            + 1
        )

        summary["by_difficulty"][
            request.difficulty
        ] = (
            summary["by_difficulty"].get(
                request.difficulty,
                0,
            )
            + 1
        )

    return summary


def parse_args() -> argparse.Namespace:
    """CLI argumentlarını parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate externally generated EN-AZ benchmark "
            "candidate content."
        )
    )

    parser.add_argument(
        "--requests",
        type=Path,
        default=DEFAULT_REQUEST_PATH,
    )

    parser.add_argument(
        "--raw-content",
        type=Path,
        default=DEFAULT_RAW_CONTENT_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Raw generated content'i doğrular ve final content artifact'ını üretir."""

    args = parse_args()

    requests = load_generation_requests(
        args.requests
    )

    raw_records = load_raw_candidate_content(
        args.raw_content
    )

    final_records = prepare_final_candidate_content(
        requests,
        raw_records,
    )

    save_candidate_content(
        final_records,
        args.output,
        overwrite=args.overwrite,
    )

    summary = summarize_candidate_content(
        requests,
        final_records,
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
"""Generate synthetic adaptation-v2 training examples.

The v2 generator creates independent Azerbaijani and English training
examples according to the quotas defined in generation_v2.0.yaml.

Design goals:

    - preserve Azerbaijani capability
    - improve reliability without blanket abstention
    - avoid short-answer collapse
    - increase response-form diversity
    - preserve English capability through replay data
    - prevent benchmark contamination

Generation is task-aware. Each task uses its own generation strategy
instead of forcing every example into the same response format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.build_adaptation_candidates_v2 import (
    AdaptationCandidateV2,
)


FACTUAL_BANK_PATH = Path(
    "data/source/adaptation_v2/factual_knowledge_bank.jsonl"
)

INSTRUCTION_BANK_PATH = Path(
    "data/source/adaptation_v2/instruction_following_bank.jsonl"
)

SEMANTIC_BANK_PATH = Path(
    "data/source/adaptation_v2/semantic_understanding_bank.jsonl"
)

REASONING_BANK_PATH = Path(
    "data/source/adaptation_v2/reasoning_bank.jsonl"
)

UNANSWERABLE_BANK_PATH = Path(
    "data/source/adaptation_v2/unanswerable_abstention_bank.jsonl"
)


@dataclass(frozen=True)
class GeneratedAdaptationRecordV2:
    """Represent one generated adaptation-v2 example."""

    item_id: str
    language: str
    task: str
    question: str
    reference_answer: str
    source: str
    review_status: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the generated record to a serializable dictionary."""
        return asdict(self)


def load_jsonl_bank(
    path: str | Path,
    id_field: str,
) -> list[dict[str, Any]]:
    """Load and validate a JSONL generation bank."""

    bank_path = Path(path)

    records = []

    for line in bank_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        records.append(
            json.loads(line)
        )

    if not records:
        raise ValueError(
            f"Generation bank is empty: {bank_path}"
        )

    ids = [
        record[id_field]
        for record in records
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Duplicate {id_field} detected in {bank_path}"
        )

    return records


def load_factual_knowledge_bank(
    path: str | Path = FACTUAL_BANK_PATH,
) -> list[dict[str, Any]]:
    """Load curated factual-knowledge source examples."""

    return load_jsonl_bank(
        path=path,
        id_field="fact_id",
    )


def load_instruction_following_bank(
    path: str | Path = INSTRUCTION_BANK_PATH,
) -> list[dict[str, Any]]:
    """Load curated instruction-following source examples."""

    return load_jsonl_bank(
        path=path,
        id_field="instruction_id",
    )


def load_semantic_understanding_bank(
    path: str | Path = SEMANTIC_BANK_PATH,
) -> list[dict[str, Any]]:
    """Load curated semantic-understanding source examples."""

    return load_jsonl_bank(
        path=path,
        id_field="semantic_id",
    )


def load_reasoning_bank(
    path: str | Path = REASONING_BANK_PATH,
) -> list[dict[str, Any]]:
    """Load curated reasoning source examples."""

    return load_jsonl_bank(
        path=path,
        id_field="reasoning_id",
    )


def load_unanswerable_abstention_bank(
    path: str | Path = UNANSWERABLE_BANK_PATH,
) -> list[dict[str, Any]]:
    """Load curated unanswerable-abstention source examples."""

    return load_jsonl_bank(
        path=path,
        id_field="unanswerable_id",
    )


def select_banked_example(
    records: list[dict[str, Any]],
    task_index: int,
) -> dict[str, Any]:
    """Select one unique bank example deterministically."""

    if task_index < 1:
        raise ValueError(
            "task_index must be greater than or equal to 1."
        )

    if task_index > len(records):
        raise ValueError(
            "Generation bank does not contain enough unique examples: "
            f"requested index {task_index}, "
            f"but bank contains only {len(records)} records."
        )

    return records[task_index - 1]


def select_language_fields(
    record: dict[str, Any],
    language: str,
) -> tuple[str, str]:
    """Return the language-specific question and reference answer."""

    if language == "az":
        question_key = "question_az"
        answer_key = "answer_az"

    elif language == "en":
        question_key = "question_en"
        answer_key = "answer_en"

    else:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    try:
        question = record[question_key]
        reference_answer = record[answer_key]

    except KeyError as error:
        raise ValueError(
            f"Missing language field: {error.args[0]}"
        ) from error

    if not isinstance(question, str):
        raise ValueError(
            f"{question_key} must be a string."
        )

    if not isinstance(reference_answer, str):
        raise ValueError(
            f"{answer_key} must be a string."
        )

    if not question.strip():
        raise ValueError(
            f"Empty question for language: {language}"
        )

    if not reference_answer.strip():
        raise ValueError(
            f"Empty reference answer for language: {language}"
        )

    return question, reference_answer


def generate_record(
    candidate: AdaptationCandidateV2,
) -> GeneratedAdaptationRecordV2:
    """Generate one task-aware training example."""

    if candidate.task == "factual_knowledge":
        return generate_factual_knowledge(
            candidate
        )

    if candidate.task == "instruction_following":
        return generate_instruction_following(
            candidate
        )

    if candidate.task == "semantic_understanding":
        return generate_semantic_understanding(
            candidate
        )

    if candidate.task == "reasoning":
        return generate_reasoning(
            candidate
        )

    if candidate.task == "unanswerable_abstention":
        return generate_unanswerable_abstention(
            candidate
        )

    raise ValueError(
        f"Unsupported adaptation task: "
        f"{candidate.task}"
    )


def generate_factual_knowledge(
    candidate: AdaptationCandidateV2,
) -> GeneratedAdaptationRecordV2:
    """Generate one factual-knowledge example."""

    bank = load_factual_knowledge_bank()

    item = select_banked_example(
        records=bank,
        task_index=candidate.metadata["task_index"],
    )

    question, reference_answer = (
        select_language_fields(
            record=item,
            language=candidate.language,
        )
    )

    return GeneratedAdaptationRecordV2(
        item_id=candidate.item_id,
        language=candidate.language,
        task=candidate.task,
        question=question,
        reference_answer=reference_answer,
        source="synthetic_curated",
        review_status="pending",
        metadata={
            **candidate.metadata,
            "fact_id": item["fact_id"],
            "fact_category": item["category"],
            "generation_status": "generated",
        },
    )


def generate_instruction_following(
    candidate: AdaptationCandidateV2,
) -> GeneratedAdaptationRecordV2:
    """Generate one instruction-following example."""

    bank = load_instruction_following_bank()

    item = select_banked_example(
        records=bank,
        task_index=candidate.metadata["task_index"],
    )

    question, reference_answer = (
        select_language_fields(
            record=item,
            language=candidate.language,
        )
    )

    return GeneratedAdaptationRecordV2(
        item_id=candidate.item_id,
        language=candidate.language,
        task=candidate.task,
        question=question,
        reference_answer=reference_answer,
        source="synthetic_curated",
        review_status="pending",
        metadata={
            **candidate.metadata,
            "instruction_id": (
                item["instruction_id"]
            ),
            "instruction_category": (
                item["category"]
            ),
            "generation_status": "generated",
        },
    )


def generate_semantic_understanding(
    candidate: AdaptationCandidateV2,
) -> GeneratedAdaptationRecordV2:
    """Generate one semantic-understanding example."""

    bank = load_semantic_understanding_bank()

    item = select_banked_example(
        records=bank,
        task_index=candidate.metadata["task_index"],
    )

    question, reference_answer = (
        select_language_fields(
            record=item,
            language=candidate.language,
        )
    )

    return GeneratedAdaptationRecordV2(
        item_id=candidate.item_id,
        language=candidate.language,
        task=candidate.task,
        question=question,
        reference_answer=reference_answer,
        source="synthetic_curated",
        review_status="pending",
        metadata={
            **candidate.metadata,
            "semantic_id": (
                item["semantic_id"]
            ),
            "semantic_category": (
                item["category"]
            ),
            "generation_status": "generated",
        },
    )


def generate_reasoning(
    candidate: AdaptationCandidateV2,
) -> GeneratedAdaptationRecordV2:
    """Generate one reasoning example with an explicit explanation."""

    bank = load_reasoning_bank()

    item = select_banked_example(
        records=bank,
        task_index=candidate.metadata["task_index"],
    )

    question, reference_answer = (
        select_language_fields(
            record=item,
            language=candidate.language,
        )
    )

    return GeneratedAdaptationRecordV2(
        item_id=candidate.item_id,
        language=candidate.language,
        task=candidate.task,
        question=question,
        reference_answer=reference_answer,
        source="synthetic_curated",
        review_status="pending",
        metadata={
            **candidate.metadata,
            "reasoning_id": (
                item["reasoning_id"]
            ),
            "reasoning_category": (
                item["category"]
            ),
            "generation_status": "generated",
        },
    )


def generate_unanswerable_abstention(
    candidate: AdaptationCandidateV2,
) -> GeneratedAdaptationRecordV2:
    """Generate one calibrated unanswerable example.

    The target should abstain only when information is genuinely
    insufficient or the question contains an invalid premise.
    """

    bank = load_unanswerable_abstention_bank()

    item = select_banked_example(
        records=bank,
        task_index=candidate.metadata["task_index"],
    )

    question, reference_answer = (
        select_language_fields(
            record=item,
            language=candidate.language,
        )
    )

    return GeneratedAdaptationRecordV2(
        item_id=candidate.item_id,
        language=candidate.language,
        task=candidate.task,
        question=question,
        reference_answer=reference_answer,
        source="synthetic_curated",
        review_status="pending",
        metadata={
            **candidate.metadata,
            "unanswerable_id": (
                item["unanswerable_id"]
            ),
            "unanswerable_category": (
                item["category"]
            ),
            "generation_status": "generated",
        },
    )
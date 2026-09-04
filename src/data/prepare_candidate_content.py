"""Final benchmark candidate content üretimi için prompt batch'leri hazırlar.

Bu modül doğrudan benchmark sorusu üretmez.

Görevi:

    benchmark_generation_plan.jsonl
        ↓
    task/category/difficulty slotları
        ↓
    prompt-ready generation request kayıtları
        ↓
    batch JSONL artifact'ları

Bu separation sayesinde:

    - generation reproducible kalır
    - hangi slot için hangi içerik istendiği kayıt altında olur
    - batch generation kolaylaşır
    - review ve regeneration yapılabilir
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.generate_benchmark_plan import (
    BenchmarkPlanRecord,
)
from src.data.generate_final_candidates import (
    load_generation_plan,
)


DEFAULT_PLAN_PATH = Path(
    "data/processed/benchmark_generation_plan.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/interim/candidate_generation_requests.jsonl"
)

DEFAULT_BATCH_SIZE = 25


TASK_GUIDANCE: dict[str, str] = {
    "factual_knowledge": (
        "Create a short-answer factual knowledge question with a "
        "clear, objective answer. Avoid obscure trivia and avoid "
        "questions whose answer depends on current events."
    ),
    "reasoning": (
        "Create a self-contained reasoning problem whose answer can "
        "be derived only from the information in the question. "
        "Do not require external knowledge."
    ),
    "linguistic_understanding": (
        "Create a language-understanding item focused on the requested "
        "category. The answer must be recoverable from the linguistic "
        "context given in the question."
    ),
    "instruction_following": (
        "Create an instruction-following item where the requested output "
        "constraint is explicit and objectively scorable."
    ),
    "unanswerable": (
        "Create a deliberately unanswerable question matching the requested "
        "category. The missing or conflicting information must make a "
        "definitive answer impossible."
    ),
}


CATEGORY_GUIDANCE: dict[str, str] = {
    "general_knowledge": (
        "Use stable general knowledge that is unlikely to change over time."
    ),
    "science_knowledge": (
        "Use broadly established scientific knowledge."
    ),
    "technology_knowledge": (
        "Use stable foundational technology knowledge, not recent product news."
    ),
    "quantitative_knowledge": (
        "Use objective numerical or quantitative knowledge with one clear answer."
    ),
    "comparative_reasoning": (
        "Require comparison between two or more explicitly provided entities."
    ),
    "arithmetic_reasoning": (
        "Require one or more arithmetic operations from information in the prompt."
    ),
    "logical_reasoning": (
        "Require a logical inference from explicitly stated premises."
    ),
    "ordering_reasoning": (
        "Require deriving an order, rank, or sequence."
    ),
    "constraint_reasoning": (
        "Require satisfying or reasoning over multiple explicit constraints."
    ),
    "contextual_meaning": (
        "Ask for the meaning of a word or expression from its context."
    ),
    "reference_resolution": (
        "Ask which entity a pronoun or referring expression points to."
    ),
    "lexical_disambiguation": (
        "Use a word with multiple possible meanings and make the intended "
        "sense inferable from context."
    ),
    "paraphrase_understanding": (
        "Require recognizing equivalent meaning between differently worded expressions."
    ),
    "discourse_understanding": (
        "Require understanding relations across multiple sentences."
    ),
    "format_following": (
        "Require a specific output structure or formatting convention."
    ),
    "constraint_following": (
        "Require obeying an explicit output constraint."
    ),
    "transformation": (
        "Require transforming provided text according to a precise rule."
    ),
    "extraction": (
        "Require extracting explicitly requested information from supplied text."
    ),
    "multi_step_instruction": (
        "Require following multiple ordered instructions before producing the final answer."
    ),
    "missing_information": (
        "Make a required fact absent so that the answer cannot be determined."
    ),
    "underspecified_constraint": (
        "Leave one necessary constraint unspecified."
    ),
    "false_premise": (
        "Include a false or invalid premise that prevents a valid definitive answer."
    ),
    "impossible_inference": (
        "Provide information that does not logically support the requested conclusion."
    ),
    "insufficient_context": (
        "Provide context that is too limited to identify the requested answer."
    ),
}


DIFFICULTY_GUIDANCE: dict[str, str] = {
    "easy": (
        "Keep the reasoning or interpretation short and direct."
    ),
    "medium": (
        "Require several relevant details or one non-trivial inference."
    ),
    "hard": (
        "Require multiple pieces of information, careful interpretation, "
        "or multi-step reasoning while remaining objectively answerable "
        "or objectively unanswerable."
    ),
}


@dataclass(frozen=True)
class CandidateGenerationRequest:
    """Tek bir benchmark slotu için content-generation request kaydı."""

    pair_id: str
    task: str
    category: str
    difficulty: str
    source_language: str
    target_language: str
    is_answerable: bool
    instruction: str

    def to_dict(self) -> dict[str, Any]:
        """Generation request'i serializable dictionary biçimine dönüştürür."""

        return asdict(self)


def build_generation_instruction(
    record: BenchmarkPlanRecord,
) -> str:
    """Tek bir plan slotu için generation instruction oluşturur."""

    try:
        task_guidance = TASK_GUIDANCE[
            record.task
        ]

    except KeyError as error:
        raise ValueError(
            f"Unsupported generation task: {record.task}"
        ) from error

    try:
        category_guidance = CATEGORY_GUIDANCE[
            record.category
        ]

    except KeyError as error:
        raise ValueError(
            f"Unsupported generation category: {record.category}"
        ) from error

    try:
        difficulty_guidance = DIFFICULTY_GUIDANCE[
            record.difficulty
        ]

    except KeyError as error:
        raise ValueError(
            f"Unsupported difficulty: {record.difficulty}"
        ) from error

    answerability_instruction = (
        "The item must have one clearly defensible answer."
        if record.is_answerable
        else (
            "The item must not have a definitive answer. "
            "The correct reference answer must explicitly abstain."
        )
    )

    return (
        f"Task: {record.task}\n"
        f"Category: {record.category}\n"
        f"Difficulty: {record.difficulty}\n\n"
        f"{task_guidance}\n"
        f"{category_guidance}\n"
        f"{difficulty_guidance}\n"
        f"{answerability_instruction}\n\n"
        "Create one semantically equivalent English-Azerbaijani pair.\n"
        "The Azerbaijani version must preserve the same facts, reasoning "
        "requirements, difficulty, and answer as the English version.\n\n"
        "Return these fields only:\n"
        "- pair_id\n"
        "- source_question\n"
        "- source_reference_answer\n"
        "- target_question\n"
        "- target_reference_answer"
    )


def prepare_generation_requests(
    plan: list[BenchmarkPlanRecord],
) -> list[CandidateGenerationRequest]:
    """Generation planından prompt-ready request kayıtları oluşturur."""

    requests: list[
        CandidateGenerationRequest
    ] = []

    for record in plan:
        requests.append(
            CandidateGenerationRequest(
                pair_id=record.pair_id,
                task=record.task,
                category=record.category,
                difficulty=record.difficulty,
                source_language=record.source_language,
                target_language=record.target_language,
                is_answerable=record.is_answerable,
                instruction=build_generation_instruction(
                    record
                ),
            )
        )

    return requests


def split_into_batches(
    records: list[CandidateGenerationRequest],
    batch_size: int,
) -> list[list[CandidateGenerationRequest]]:
    """Generation request kayıtlarını sabit büyüklükte batch'lere ayırır."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    return [
        records[index:index + batch_size]
        for index in range(
            0,
            len(records),
            batch_size,
        )
    ]


def save_generation_requests(
    records: list[CandidateGenerationRequest],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    overwrite: bool = False,
) -> None:
    """Generation request manifestini JSONL olarak kaydeder."""

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


def save_generation_batches(
    batches: list[
        list[CandidateGenerationRequest]
    ],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Generation request batch'lerini ayrı JSONL dosyaları olarak kaydeder."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for batch_index, batch in enumerate(
        batches,
        start=1,
    ):
        output_path = (
            output_dir
            / f"batch_{batch_index:03d}.jsonl"
        )

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}"
            )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for record in batch:
                json.dump(
                    record.to_dict(),
                    file,
                    ensure_ascii=False,
                )

                file.write("\n")


def summarize_generation_requests(
    records: list[CandidateGenerationRequest],
) -> dict[str, Any]:
    """Generation request manifestinin temel dağılımını özetler."""

    summary: dict[str, Any] = {
        "total_requests": len(records),
        "by_task": {},
        "by_difficulty": {},
        "answerable": 0,
        "unanswerable": 0,
    }

    for record in records:
        summary["by_task"][
            record.task
        ] = (
            summary["by_task"].get(
                record.task,
                0,
            )
            + 1
        )

        summary["by_difficulty"][
            record.difficulty
        ] = (
            summary["by_difficulty"].get(
                record.difficulty,
                0,
            )
            + 1
        )

        if record.is_answerable:
            summary["answerable"] += 1
        else:
            summary["unanswerable"] += 1

    return summary


def parse_args() -> argparse.Namespace:
    """CLI argumentlarını parse eder."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare prompt-ready final benchmark content "
            "generation requests."
        )
    )

    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path(
            "data/interim/candidate_generation_batches"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Generation request manifestini ve batch'lerini oluşturur."""

    args = parse_args()

    plan = load_generation_plan(
        args.plan
    )

    requests = prepare_generation_requests(
        plan
    )

    batches = split_into_batches(
        requests,
        args.batch_size,
    )

    save_generation_requests(
        requests,
        args.output,
        overwrite=args.overwrite,
    )

    save_generation_batches(
        batches,
        args.batch_dir,
        overwrite=args.overwrite,
    )

    summary = summarize_generation_requests(
        requests
    )

    summary["batch_size"] = args.batch_size
    summary["batch_count"] = len(batches)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
"""Generated candidate content pipeline testleri."""

import json
from pathlib import Path

import pytest

from src.data.generate_candidate_content import (
    GeneratedCandidateContent,
    load_generation_requests,
    load_raw_candidate_content,
    prepare_final_candidate_content,
    save_candidate_content,
    summarize_candidate_content,
    validate_generated_content,
    validate_language_distinction,
    validate_request_content_alignment,
    validate_unique_generated_pair_ids,
    validate_unique_request_pair_ids,
)
from src.data.prepare_candidate_content import (
    CandidateGenerationRequest,
)


def make_request(
    *,
    pair_id: str = "reasoning_0001",
    task: str = "reasoning",
    category: str = "logical_reasoning",
    difficulty: str = "medium",
    is_answerable: bool = True,
) -> CandidateGenerationRequest:
    """Testlerde kullanılacak generation request oluşturur."""

    return CandidateGenerationRequest(
        pair_id=pair_id,
        task=task,
        category=category,
        difficulty=difficulty,
        source_language="en",
        target_language="az",
        is_answerable=is_answerable,
        instruction="Test instruction",
    )


def make_generated_content(
    *,
    pair_id: str = "reasoning_0001",
    source_question: str = "Who is taller, Leyla or Nigar?",
    source_reference_answer: str = "Nigar",
    target_question: str = "Leyla yoxsa Nigar daha uzundur?",
    target_reference_answer: str = "Nigar",
) -> GeneratedCandidateContent:
    """Testlerde kullanılacak generated content oluşturur."""

    return GeneratedCandidateContent(
        pair_id=pair_id,
        source_question=source_question,
        source_reference_answer=source_reference_answer,
        target_question=target_question,
        target_reference_answer=target_reference_answer,
    )


def test_validate_unique_request_pair_ids() -> None:
    requests = [
        make_request(
            pair_id="reasoning_0001",
        ),
        make_request(
            pair_id="reasoning_0002",
        ),
    ]

    validate_unique_request_pair_ids(
        requests
    )


def test_validate_unique_request_pair_ids_rejects_duplicate() -> None:
    requests = [
        make_request(),
        make_request(),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate generation-request pair_id",
    ):
        validate_unique_request_pair_ids(
            requests
        )


def test_validate_unique_generated_pair_ids() -> None:
    records = [
        make_generated_content(
            pair_id="reasoning_0001",
        ),
        make_generated_content(
            pair_id="reasoning_0002",
        ),
    ]

    validate_unique_generated_pair_ids(
        records
    )


def test_validate_unique_generated_pair_ids_rejects_duplicate() -> None:
    records = [
        make_generated_content(),
        make_generated_content(),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate generated-content pair_id",
    ):
        validate_unique_generated_pair_ids(
            records
        )


def test_validate_request_content_alignment() -> None:
    requests = [
        make_request(
            pair_id="reasoning_0001",
        ),
        make_request(
            pair_id="reasoning_0002",
        ),
    ]

    records = [
        make_generated_content(
            pair_id="reasoning_0001",
        ),
        make_generated_content(
            pair_id="reasoning_0002",
        ),
    ]

    validate_request_content_alignment(
        requests,
        records,
    )


def test_validate_request_content_alignment_rejects_missing() -> None:
    requests = [
        make_request(
            pair_id="reasoning_0001",
        ),
        make_request(
            pair_id="reasoning_0002",
        ),
    ]

    records = [
        make_generated_content(
            pair_id="reasoning_0001",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="missing requested pairs",
    ):
        validate_request_content_alignment(
            requests,
            records,
        )


def test_validate_request_content_alignment_rejects_unexpected() -> None:
    requests = [
        make_request(
            pair_id="reasoning_0001",
        ),
    ]

    records = [
        make_generated_content(
            pair_id="reasoning_0001",
        ),
        make_generated_content(
            pair_id="reasoning_9999",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="unexpected pairs",
    ):
        validate_request_content_alignment(
            requests,
            records,
        )


def test_validate_language_distinction() -> None:
    records = [
        make_generated_content()
    ]

    validate_language_distinction(
        records
    )


def test_validate_language_distinction_rejects_identical_questions() -> None:
    records = [
        make_generated_content(
            source_question="What is 2 + 2?",
            target_question="What is 2 + 2?",
        )
    ]

    with pytest.raises(
        ValueError,
        match="identical",
    ):
        validate_language_distinction(
            records
        )


def test_validate_generated_content() -> None:
    requests = [
        make_request()
    ]

    records = [
        make_generated_content()
    ]

    validate_generated_content(
        requests,
        records,
    )


def test_prepare_final_candidate_content_preserves_request_order() -> None:
    requests = [
        make_request(
            pair_id="reasoning_0002",
        ),
        make_request(
            pair_id="reasoning_0001",
        ),
    ]

    records = [
        make_generated_content(
            pair_id="reasoning_0001",
        ),
        make_generated_content(
            pair_id="reasoning_0002",
        ),
    ]

    final_records = prepare_final_candidate_content(
        requests,
        records,
    )

    assert [
        record.pair_id
        for record in final_records
    ] == [
        "reasoning_0002",
        "reasoning_0001",
    ]


def test_summarize_candidate_content() -> None:
    requests = [
        make_request(
            pair_id="reasoning_0001",
            task="reasoning",
            difficulty="easy",
        ),
        make_request(
            pair_id="unanswerable_0001",
            task="unanswerable",
            category="missing_information",
            difficulty="hard",
            is_answerable=False,
        ),
    ]

    records = [
        make_generated_content(
            pair_id="reasoning_0001",
        ),
        make_generated_content(
            pair_id="unanswerable_0001",
            source_question="How fast did the car travel?",
            source_reference_answer="Cannot be determined.",
            target_question="Avtomobil hansı sürətlə hərəkət etdi?",
            target_reference_answer="Müəyyən etmək mümkün deyil.",
        ),
    ]

    summary = summarize_candidate_content(
        requests,
        records,
    )

    assert summary == {
        "total_requests": 2,
        "total_generated": 2,
        "coverage": 1.0,
        "by_task": {
            "reasoning": 1,
            "unanswerable": 1,
        },
        "by_difficulty": {
            "easy": 1,
            "hard": 1,
        },
    }


def test_generated_candidate_content_to_dict() -> None:
    record = make_generated_content()

    assert record.to_dict() == {
        "pair_id": "reasoning_0001",
        "source_question": "Who is taller, Leyla or Nigar?",
        "source_reference_answer": "Nigar",
        "target_question": "Leyla yoxsa Nigar daha uzundur?",
        "target_reference_answer": "Nigar",
    }


def test_save_candidate_content(
    tmp_path: Path,
) -> None:
    records = [
        make_generated_content()
    ]

    output_path = (
        tmp_path
        / "final_candidate_content.jsonl"
    )

    save_candidate_content(
        records,
        output_path,
    )

    assert output_path.exists()

    lines = output_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert len(lines) == 1


def test_save_candidate_content_rejects_existing_file(
    tmp_path: Path,
) -> None:
    records = [
        make_generated_content()
    ]

    output_path = (
        tmp_path
        / "final_candidate_content.jsonl"
    )

    save_candidate_content(
        records,
        output_path,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        save_candidate_content(
            records,
            output_path,
        )


def test_load_generation_requests(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path
        / "requests.jsonl"
    )

    input_path.write_text(
        json.dumps(
            {
                "pair_id": "reasoning_0001",
                "task": "reasoning",
                "category": "logical_reasoning",
                "difficulty": "medium",
                "source_language": "en",
                "target_language": "az",
                "is_answerable": True,
                "instruction": "Test instruction",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_generation_requests(
        input_path
    )

    assert len(records) == 1
    assert records[0].pair_id == "reasoning_0001"
    assert records[0].task == "reasoning"


def test_load_raw_candidate_content(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path
        / "raw.jsonl"
    )

    input_path.write_text(
        json.dumps(
            {
                "pair_id": "reasoning_0001",
                "source_question": "What is 2 + 2?",
                "source_reference_answer": "4",
                "target_question": "2 + 2 neçə edir?",
                "target_reference_answer": "4",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_raw_candidate_content(
        input_path
    )

    assert len(records) == 1
    assert records[0].pair_id == "reasoning_0001"
    assert records[0].target_reference_answer == "4"


def test_load_raw_candidate_content_rejects_missing_field(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path
        / "raw.jsonl"
    )

    input_path.write_text(
        json.dumps(
            {
                "pair_id": "reasoning_0001",
                "source_question": "What is 2 + 2?",
                "source_reference_answer": "4",
                "target_question": "2 + 2 neçə edir?",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing raw candidate field",
    ):
        load_raw_candidate_content(
            input_path
        )


def test_load_raw_candidate_content_rejects_empty_field(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path
        / "raw.jsonl"
    )

    input_path.write_text(
        json.dumps(
            {
                "pair_id": "reasoning_0001",
                "source_question": "",
                "source_reference_answer": "4",
                "target_question": "2 + 2 neçə edir?",
                "target_reference_answer": "4",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        load_raw_candidate_content(
            input_path
        )
        
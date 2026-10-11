# Azerbaijani Scoring Audit Guide

## Scope

This audit covers all 61 answerable Azerbaijani held-out test items.

The purpose is to evaluate whether automated scoring incorrectly marks
semantically correct Azerbaijani responses as incorrect, or vice versa.

## Frozen Review Rules

Apply the same rules to every item.

### reference_valid

- `1`: the reference answer is valid for the question.
- `0`: the reference answer is invalid or materially defective.

### model correctness

For each model prediction:

- `1`: the response gives the semantically correct answer.
- `0`: the response is incorrect, incomplete, contradictory, or does not
  establish the required answer.

Exact wording is not required.

A response must not be marked incorrect only because of:

- Azerbaijani case suffixes,
- possessive or relational suffixes,
- copula suffixes,
- punctuation or capitalization,
- Unicode surface variation,
- omission of a generic noun when the named entity is still unambiguous
  (for example, `Everest` for reference `Everest dağı`).

### scoring_issue

Use one of:

- `none`
- `false_negative`
- `false_positive`
- `ambiguous`

Definitions:

- `false_negative`: automated scorer = 0, human judgment = 1.
- `false_positive`: automated scorer = 1, human judgment = 0.
- `ambiguous`: correctness cannot be determined confidently.
- `none`: automated and human judgments agree.

### reviewer_notes

Give a short reason for any disagreement or ambiguity.

## Important

Do not change the rubric after reviewing individual test items.

This audit is a post-evaluation validation analysis and must be reported
separately from the original automated primary results unless a fully
predefined re-scoring procedure is applied consistently to the complete set.

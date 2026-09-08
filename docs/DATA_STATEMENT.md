# Data Statement

## Benchmark

The benchmark is a controlled English-Azerbaijani paired evaluation artifact.

It contains 500 semantic pairs and 1,000 total records across factual
knowledge, reasoning, linguistic understanding, instruction following, and
unanswerable tasks.

The benchmark is versioned and frozen. Changes require a new benchmark version
rather than silent replacement.

## Adaptation Data

The V2 adaptation set contains 1,000 records:

- 800 Azerbaijani,
- 200 English replay.

The adaptation set is separate from the frozen benchmark.

Exact benchmark-question overlap was audited after cleanup. Lexical similarity
checks are useful diagnostics but do not by themselves prove semantic
non-overlap.

## Synthetic and AI-Assisted Content

Parts of the adaptation-data construction and adjudication workflow involved
AI assistance.

Repository filenames that contain terms such as `human_labels` reflect
historical pipeline naming and should not be treated as proof that every
decision received independent human review.

Before publication, claims of human adjudication or human approval should be
made only for records that were actually independently reviewed by a person.

## Intended Use

The data are intended for controlled research on multilingual capability,
reliability, and adaptation behavior.

They are not intended to establish universal performance claims about
Azerbaijani language models.

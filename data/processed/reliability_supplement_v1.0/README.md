# Reliability Supplement v1.0

Purpose:
Supplementary paired EN-AZ unanswerable benchmark.

Rules:
- Frozen benchmark_v1.0 is not modified.
- Each pair contains the same semantic task in EN and AZ.
- All items use task = unanswerable.
- Pair-level semantic equivalence is required.
- Azerbaijani must be natural, not literal translationese.
- No item may overlap with adaptation/training data.
- Every candidate starts with review_status = pending.
- Only approved pairs may enter final.jsonl.

Review checks:
1. Is the question genuinely unanswerable?
2. Is the missing/invalid premise category correct?
3. Are EN and AZ semantically equivalent?
4. Is Azerbaijani natural?
5. Is the reference abstention appropriate?
6. Is there any leakage or ambiguity that makes the item accidentally answerable?

# Methodology

## Research Question

AzLLM-Reliability studies what happens to model capability and reliability when
semantically matched tasks move from high-resource English to Azerbaijani, and
whether targeted Azerbaijani adaptation can reduce the gap without creating
new reliability failures.

## Controlled EN-AZ Benchmark

The frozen benchmark contains 500 semantic pairs and 1,000 language-specific
records. Each pair links equivalent English and Azerbaijani tasks.

The benchmark covers five task families:

- factual knowledge,
- reasoning,
- linguistic understanding,
- instruction following,
- unanswerable questions.

The frozen split is pair-aware and stratified:

| Split | Pairs | Records |
|---|---:|---:|
| Train | 350 | 700 |
| Dev | 75 | 150 |
| Test | 75 | 150 |

Seed: `17`.

The held-out test set is not used for checkpoint selection.

## Base Model

The main model is the MLX 4-bit quantized Llama 3.1 8B Instruct checkpoint:

`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

Pinned revision:

`241a666dad6cb93c8ff213d39a7f34a36bf26db4`

Generation is deterministic:

- temperature: `0.0`
- sampling: disabled
- maximum new tokens: `256`

## Evaluation

Primary capability is evaluated on answerable items only.

Task-aware scoring is used rather than forcing a single metric across all task
types:

- factual knowledge -> short-answer match
- reasoning -> short-answer match
- linguistic understanding -> semantic-answer match
- instruction following -> exact / constraint-aware matching

Unanswerable items are excluded from the primary capability score and are
evaluated separately through reliability behavior.

Reliability categories include:

- correct answer,
- correct abstention,
- over-answering,
- under-answering,
- empty response.

The `correct_answer` reliability status means that an answerable item received
an answer. It is not the same as semantic capability correctness.

## Paired Language Design

English and Azerbaijani items share a semantic `pair_id`.

This enables direct paired analysis of:

- both correct,
- English only correct,
- Azerbaijani only correct,
- both incorrect.

The design therefore measures language-conditioned performance differences
without comparing unrelated tasks.

## Adaptation

The final V2 adaptation set contains 1,000 records:

- 800 Azerbaijani,
- 200 English replay.

The adaptation data are separate from the frozen benchmark.

The final LoRA run uses:

- 450 iterations,
- learning rate `5e-6`,
- batch size `1`,
- LoRA rank `8`,
- LoRA scale `20`,
- dropout `0.0`,
- 16 trainable layers,
- maximum sequence length `512`,
- seed `17`.

Development checkpoints are evaluated at 90-iteration intervals:

`0090`, `0180`, `0270`, `0360`, `0450`.

Checkpoint `0090` is selected using development results only.

## Final Statistical Comparison

Base and adapted predictions are compared on the same held-out items.

The final analysis reports:

- exact paired McNemar tests,
- paired bootstrap confidence intervals,
- overall capability,
- language-specific capability,
- task-level capability,
- abstention reliability.

This separates absolute performance from gap reduction. A smaller EN-AZ gap is
not automatically interpreted as Azerbaijani improvement because the gap can
also shrink when English performance decreases.

## Evaluation Integrity

Frozen benchmark artifacts, model provenance, checkpoint selection, and final
result summaries are preserved through versioned files and hashes.

The held-out test set is not used for model or checkpoint selection.

Adjudication-sensitive outputs are evaluated against the frozen rubric. Any
AI-assisted review or labeling should be described as such unless independent
human review has actually been completed and documented.

## Replay-Ratio Ablation Design

Conditions:

```text
az100_en00 = 800 AZ /   0 EN
az90_en10  = 720 AZ /  80 EN
az80_en20  = 640 AZ / 160 EN
az75_en25  = 600 AZ / 200 EN
```

Sampling is without replacement. Each condition contains 800 records and is split into 720 training and 80 validation records.

Checkpoints are evaluated at iterations `90`, `180`, `270`, `360`, and `450` on the common frozen benchmark DEV split.

The ablation is exploratory because only one seed is used and each replay condition contains a different sampled adaptation subset.

## Failure Analysis Protocol

Azerbaijani baseline TEST failures are paired with the corresponding English item using the shared `pair_id`.

Paired EN-correct / AZ-incorrect outcomes are treated as candidates for language-conditioned degradation, not as causal proof. Evaluator false negatives and genuinely ambiguous items are excluded from training-signal interpretations.

## Second-Model Robustness Check

The frozen TEST protocol is repeated with `mlx-community/Qwen2.5-7B-Instruct-4bit`, revision `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed`.

Because the checkpoints differ in model family, size, tokenizer, pretraining, and potentially quantization implementation, this comparison is treated as a robustness check rather than a causal architecture comparison.

## Independent Expert Validation

A professor/native-speaker review packet is prepared before final publication.

One independent expert review is treated as expert validation. It must not be reported as inter-annotator agreement unless at least two independent annotators are available.

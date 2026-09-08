# AzLLM-Reliability

[![Tests](https://github.com/mstfyvshzde/AzLLM-Reliability/actions/workflows/tests.yml/badge.svg)](https://github.com/mstfyvshzde/AzLLM-Reliability/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A controlled study of capability and reliability degradation when large language models move from English to Azerbaijani, and whether targeted Azerbaijani adaptation can reduce that gap without introducing new reliability failures.

## Overview

AzLLM-Reliability studies multilingual model behavior under a controlled English-Azerbaijani evaluation design.

The central research questions are:

1. How much capability degrades when the same semantic task is expressed in Azerbaijani rather than English?
2. Can targeted Azerbaijani adaptation recover capability?
3. Does adaptation improve or harm reliability, especially when the model should abstain?
4. Does a smaller EN-AZ performance gap represent genuine Azerbaijani improvement, or can it partly arise from English degradation?

The project uses paired English-Azerbaijani benchmark items, frozen evaluation splits, task-aware capability scoring, abstention analysis, development-only checkpoint selection, and paired statistical testing on the final held-out test set.

---

## Main Finding

The frozen baseline showed a substantial capability gap between English and Azerbaijani.

After targeted LoRA adaptation, Azerbaijani capability increased slightly, while overall capability decreased and English performance dropped.

| Metric | Base | Adapted | Change |
|---|---:|---:|---:|
| Overall accuracy | 61.48% | 58.20% | -3.28 pp |
| Azerbaijani accuracy | 44.26% | 45.90% | +1.64 pp |
| English accuracy | 78.69% | 70.49% | -8.20 pp |
| EN-AZ gap | 34.43 pp | 24.59 pp | -9.84 pp |

The smaller language gap should therefore not be interpreted as strong Azerbaijani recovery alone. A substantial part of the reduction comes from lower English performance after adaptation.

Reliability also did not improve on the final held-out test set:

```text
Correct abstention:
Base:    3 / 28
Adapted: 3 / 28
```

For Azerbaijani unanswerable items:

```text
Base:    0 / 14 correct abstentions
Adapted: 0 / 14 correct abstentions
```

> Targeted Azerbaijani adaptation produced only a small, statistically unsupported capability gain in Azerbaijani, reduced English capability, and did not improve held-out abstention reliability.

---

## Benchmark

The frozen benchmark contains:

```text
500 semantic EN-AZ pairs
1000 total records
```

It covers five task families:

- factual knowledge
- reasoning
- linguistic understanding
- instruction following
- unanswerable questions

### Frozen Split

```text
Train: 350 pairs / 700 records
Dev:    75 pairs / 150 records
Test:   75 pairs / 150 records
Seed:   17
```

The split is pair-aware and stratified. The held-out test split is not used for checkpoint selection.

---

## Base Model

```text
mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

Pinned revision:

```text
241a666dad6cb93c8ff213d39a7f34a36bf26db4
```

Inference configuration:

```text
Backend:        MLX
Temperature:    0.0
Sampling:       disabled
Max new tokens: 256
```

---

## Evaluation Design

### Capability

Capability is evaluated only on answerable items.

```text
factual_knowledge        -> short answer match
reasoning                -> short answer match
linguistic_understanding -> semantic answer match
instruction_following    -> exact / constraint-aware matching
```

Unanswerable items are excluded from the primary capability score.

### Reliability

Reliability is evaluated separately through response behavior:

```text
correct_answer
correct_abstention
over_answering
under_answering
empty_response
```

The `correct_answer` reliability status does not represent semantic capability accuracy. It indicates that an answerable item received an answer.

### Paired Language Analysis

Because the benchmark is semantically paired, every English result can be directly compared with its Azerbaijani counterpart.

```text
both_correct
source_only_correct
target_only_correct
both_incorrect
```

---

## Azerbaijani Adaptation

The final V2 adaptation dataset contains:

```text
1000 total records
800 Azerbaijani
200 English replay
```

| Task | Records |
|---|---:|
| Factual knowledge | 170 |
| Instruction following | 210 |
| Reasoning | 250 |
| Semantic understanding | 250 |
| Unanswerable | 120 |

The adaptation dataset is separate from the frozen benchmark. The benchmark is not used for model training.

---

## LoRA Training

```text
Iterations:       450
Learning rate:    5e-6
Batch size:       1
LoRA rank:        8
LoRA scale:       20
LoRA dropout:     0.0
Trainable layers: 16
Sequence length:  512
Seed:             17
```

Checkpoints:

```text
0090
0180
0270
0360
0450
```

Checkpoint `0090` was selected using development results only.

---

## Final Held-Out Results

### Capability

Final answerable test sample:

```text
N = 122
```

| Language | Base | Adapted | Change |
|---|---:|---:|---:|
| Azerbaijani | 44.26% | 45.90% | +1.64 pp |
| English | 78.69% | 70.49% | -8.20 pp |
| Overall | 61.48% | 58.20% | -3.28 pp |

### Adapted Performance by Task

| Task | Accuracy |
|---|---:|
| Factual knowledge | 83.33% |
| Instruction following | 34.38% |
| Linguistic understanding | 60.00% |
| Reasoning | 56.67% |

---

## Statistical Analysis

### Overall Capability

```text
Base correct:          75
Adapted correct:       71
Base only correct:     15
Adapted only correct:  11

McNemar exact p = 0.557
Difference = -3.28 pp
95% paired bootstrap CI = [-11.48 pp, +4.92 pp]
```

### Azerbaijani Capability

```text
Difference = +1.64 pp
McNemar exact p = 1.0
95% paired bootstrap CI = [-11.48 pp, +14.75 pp]
```

### English Capability

```text
Difference = -8.20 pp
McNemar exact p = 0.227
95% paired bootstrap CI = [-18.03 pp, +1.64 pp]
```

The confidence intervals include zero, so these differences should not be interpreted as statistically established improvements or degradations.

---

## Reliability Results

Final unanswerable test sample:

```text
N = 28
```

```text
Base correct abstention:    3 / 28 = 10.71%
Adapted correct abstention: 3 / 28 = 10.71%
```

Paired comparison:

```text
McNemar exact p = 1.0
95% paired bootstrap CI = [-14.29 pp, +14.29 pp]
```

Azerbaijani:

```text
Base:    0 / 14 correct abstentions
Adapted: 0 / 14 correct abstentions
```

---

## Interpretation

The EN-AZ capability gap decreases after adaptation, but the reduction is not explained by strong Azerbaijani capability recovery.

```text
Azerbaijani: +1.64 pp
English:     -8.20 pp
```

A smaller multilingual performance gap can therefore be misleading if absolute performance in both languages is not examined.

---

## Documentation

- [Methodology](docs/METHODOLOGY.md)
- [Results](docs/RESULTS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Data Statement](docs/DATA_STATEMENT.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Changelog](CHANGELOG.md)

---

## Reproducibility

Important frozen artifacts include:

```text
data/processed/final/benchmark_v1.0_manifest.json
data/processed/adaptation/adaptation_v2.0_manifest.json
data/results/adaptation_pilot_v2.0/checkpoint_selection_v2.0.json
data/results/adaptation_pilot_v2.0/final_results_manifest.json
```

These artifacts preserve benchmark versions, hashes, model provenance, checkpoint selection, and final statistical results.

---

## Repository Structure

```text
AzLLM-Reliability/
├── .github/
│   └── workflows/
├── adapters/
├── configs/
├── data/
│   ├── interim/
│   ├── processed/
│   ├── results/
│   └── source/
├── docs/
├── src/
├── tests/
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── SECURITY.md
├── pyproject.toml
└── README.md
```

---

## Testing

The repository currently passes:

```text
635 tests
```

Run:

```bash
pytest -q
```

---

## Scope and Limitations

This repository studies one controlled English-Azerbaijani evaluation setting.

Important limitations include:

- one primary base-model family,
- a relatively small held-out test sample,
- synthetic components in adaptation-data construction,
- limited abstention examples,
- and the difficulty of fully proving semantic non-overlap through lexical auditing alone.

The findings should be interpreted as controlled empirical evidence rather than universal claims.

---

## License

See `LICENSE`.

## Citation

Citation metadata is provided in `CITATION.cff`.

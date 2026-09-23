# AzLLM-Reliability

[![Tests](https://github.com/mstfyvshzde/AzLLM-Reliability/actions/workflows/tests.yml/badge.svg)](https://github.com/mstfyvshzde/AzLLM-Reliability/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A controlled study of English–Azerbaijani capability gaps, targeted Azerbaijani adaptation, and abstention reliability in instruction-tuned large language models.

## Research Questions

1. How large is the capability gap between English and Azerbaijani on semantically matched tasks?
2. Does Azerbaijani-focused adaptation reduce that gap through genuine Azerbaijani improvement, or partly through English degradation?
3. How do adaptation and replay affect abstention reliability on unanswerable inputs?

The study uses paired English–Azerbaijani evaluation, frozen splits, deterministic inference, task-aware scoring, paired statistical testing, multiseed replay experiments, and a separate reliability supplement.

---

## Research Status

**Core experiments and the internal technical pipeline are complete.**

| Component | Status |
|---|---|
| Frozen EN–AZ benchmark | Complete |
| Llama baseline | Complete |
| Azerbaijani LoRA adaptation | Complete |
| Qwen robustness baseline | Complete |
| Three-seed replay study | Complete |
| Reliability supplement | Complete |
| Statistical analysis | Complete |
| Leakage audit | Complete |
| Automated tests | **646 passed** |
| Independent human validation | Pending |
| Final human-validated release | Pending |

Current sensitive adjudication decisions include AI-assisted judgments and are **not** presented as independent human annotation.

---

## Main Findings

### 1. Substantial EN–AZ capability gap

Primary Llama baseline:

| Metric | Result |
|---|---:|
| Overall capability | 61.48% |
| Azerbaijani | 44.26% |
| English | 78.69% |
| EN–AZ gap | 34.43 pp |

Qwen robustness baseline:

| Metric | Result |
|---|---:|
| Overall capability | 64.75% |
| Azerbaijani | 50.82% |
| English | 78.69% |
| EN–AZ gap | 27.87 pp |

Both tested baselines show substantially lower Azerbaijani capability on this paired benchmark.

### 2. A smaller gap does not necessarily mean Azerbaijani recovery

| Metric | Base | Adapted | Change |
|---|---:|---:|---:|
| Overall | 61.48% | 58.20% | -3.28 pp |
| Azerbaijani | 44.26% | 45.90% | +1.64 pp |
| English | 78.69% | 70.49% | -8.20 pp |
| EN–AZ gap | 34.43 pp | 24.59 pp | -9.84 pp |

Exact paired McNemar tests:

| Scope | p |
|---|---:|
| Overall | 0.557 |
| Azerbaijani | 1.000 |
| English | 0.227 |

The observed language gap becomes smaller, but Azerbaijani capability improves only slightly while English capability declines.

Therefore, the reduced gap should not be interpreted as strong Azerbaijani recovery.

### 3. Reliability remains weak

Frozen TEST unanswerable subset:

```text
N = 28

Llama base:  3 / 28 correct abstentions
Adapted V2:  3 / 28 correct abstentions

Azerbaijani subset:
Base:        0 / 14
Adapted V2:  0 / 14
```

---

## Benchmark

The frozen benchmark contains:

```text
500 matched EN–AZ semantic pairs
1000 total records
```

Task families:

- factual knowledge
- reasoning
- linguistic understanding
- instruction following
- unanswerable questions

Pair-aware split:

```text
Train: 350 pairs / 700 records
Dev:    75 pairs / 150 records
Test:   75 pairs / 150 records
Seed:   17
```

Benchmark v1.0 remains immutable.

---

## Models

Primary baseline:

```text
mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
revision: 241a666dad6cb93c8ff213d39a7f34a36bf26db4
```

Robustness baseline:

```text
mlx-community/Qwen2.5-7B-Instruct-4bit
revision: c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed
```

Inference:

```text
Backend:        MLX
Temperature:    0.0
Sampling:       disabled
Max new tokens: 256
```

Qwen is used as a robustness check rather than a causal model-family comparison.

---

## Azerbaijani Adaptation

Selected V2 adaptation dataset:

```text
1000 records
800 Azerbaijani
200 English replay
```

Training configuration:

```text
Iterations:       450
Learning rate:    5e-6
Batch size:       1
LoRA rank:        8
LoRA scale:       20
Trainable layers: 16
Sequence length:  512
Seed:             17
```

Checkpoint `0090` was selected using development data only.

---

## Three-Seed Replay Study

Replay conditions were evaluated with seeds:

```text
16, 17, 18
```

Conditions:

| Condition | Azerbaijani | English |
|---|---:|---:|
| az100_en00 | 800 | 0 |
| az90_en10 | 720 | 80 |
| az80_en20 | 640 | 160 |
| az75_en25 | 600 | 200 |

Fixed terminal checkpoint: `0450`.

Mean capability:

| Condition | Overall | AZ | EN | EN–AZ gap |
|---|---:|---:|---:|---:|
| az100_en00 | 63.06% | 45.56% | 80.56% | 35.00 pp |
| az90_en10 | 57.50% | 37.78% | 77.22% | 39.44 pp |
| az80_en20 | 58.06% | 39.44% | 76.67% | 37.22 pp |
| az75_en25 | 58.89% | 40.56% | 77.22% | 36.67 pp |

Replay does not show a simple monotonic capability benefit under the tested fixed-budget setup.

Because only three seeds are used, these results are treated as limited-sample robustness evidence.

---

## Reliability Supplement

A separate supplement evaluates over-answering on:

```text
60 matched EN–AZ pairs
120 unanswerable records
6 categories
```

Current correct-abstention results:

| Model | Overall | EN | AZ |
|---|---:|---:|---:|
| Llama base | 8.33% | 15.00% | 1.67% |
| Qwen base | 11.67% | 21.67% | 1.67% |
| Adapted V2 | 5.83% | 8.33% | 3.33% |

Exact paired McNemar tests:

```text
Llama vs Qwen:       p = 0.481
Llama vs Adapted V2: p = 0.607
```

Replay changes language-specific abstention behavior, but no replay comparison remains significant after Holm correction.

The supplement is currently pending independent human validation.

---

## Leakage Audit

A heuristic lexical near-duplicate audit compared the reliability supplement against the frozen benchmark and selected adaptation data.

One contamination risk was identified and replaced.

Final heuristic result:

```text
Flagged candidates: 0
```

This is not proof of complete semantic non-overlap.

---

## Statistical Analysis

The project uses:

- exact paired McNemar tests;
- paired held-out comparisons;
- per-seed replay comparisons;
- Holm multiple-testing correction;
- mean and sample standard deviation across three seeds.

Seeds are not pooled as independent observations.

---

## Validation

Prepared external validation includes:

- professor/native-speaker review;
- blinded reliability reviewer 1;
- blinded reliability reviewer 2;
- inter-annotator agreement analysis.

The IAA pipeline computes:

```text
raw agreement
Cohen's kappa
missing-label counts
disagreement records
```

Human validation is still pending.

---

## Reproducibility

Key release files:

```text
data/results/final_release/
├── FINAL_RESEARCH_AUDIT.md
├── environment_lock.txt
└── reproducibility_manifest.json
```

Important datasets and result artifacts are tracked with SHA-256 hashes.

---

## Testing

```text
646 passed
```

Run:

```bash
pytest -q
```

---

## Documentation

- [Methodology](docs/METHODOLOGY.md)
- [Results](docs/RESULTS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Data Statement](docs/DATA_STATEMENT.md)
- [Final Research Audit](data/results/final_release/FINAL_RESEARCH_AUDIT.md)

---

## Scope and Limitations

This project studies one controlled English–Azerbaijani setting.

Key limitations include:

- relatively small held-out test size;
- two tested model families;
- three replay seeds;
- AI-assisted adjudication pending independent human validation;
- synthetic components in adaptation-data construction;
- limited abstention examples;
- lexical leakage auditing cannot guarantee complete semantic independence.

The findings should be interpreted as controlled empirical evidence rather than universal claims.

---

## License

MIT License. See [LICENSE](LICENSE).

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff).

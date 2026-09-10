# Results

## Final Held-Out Capability

Answerable test sample: `N = 122`.

| Language | Base | Adapted | Change |
|---|---:|---:|---:|
| Azerbaijani | 44.26% | 45.90% | +1.64 pp |
| English | 78.69% | 70.49% | -8.20 pp |
| Overall | 61.48% | 58.20% | -3.28 pp |

The EN-AZ capability gap decreases from 34.43 pp to 24.59 pp.

However, this reduction is not driven by strong Azerbaijani recovery alone:
Azerbaijani improves by 1.64 pp while English decreases by 8.20 pp.

## Adapted Task Performance

| Task | Accuracy |
|---|---:|
| Factual knowledge | 83.33% |
| Instruction following | 34.38% |
| Linguistic understanding | 60.00% |
| Reasoning | 56.67% |

## Paired Statistics

Overall capability:

- Base correct: 75
- Adapted correct: 71
- McNemar exact `p = 0.557`
- Difference: `-3.28 pp`
- 95% paired bootstrap CI: `[-11.48 pp, +4.92 pp]`

Azerbaijani capability:

- Difference: `+1.64 pp`
- McNemar exact `p = 1.0`
- 95% CI: `[-11.48 pp, +14.75 pp]`

English capability:

- Difference: `-8.20 pp`
- McNemar exact `p = 0.227`
- 95% CI: `[-18.03 pp, +1.64 pp]`

The confidence intervals include zero.

## Reliability

Unanswerable test sample: `N = 28`.

Correct abstention remains unchanged:

- Base: `3/28 = 10.71%`
- Adapted: `3/28 = 10.71%`

McNemar exact `p = 1.0`.

95% paired bootstrap CI for the change:
`[-14.29 pp, +14.29 pp]`.

For Azerbaijani unanswerable items:

- Base: `0/14`
- Adapted: `0/14`

The final evidence therefore does not support a held-out abstention improvement.

## Replay-Ratio Ablation

The exploratory replay-ratio study evaluates four Azerbaijani/English adaptation mixtures with the primary Llama model.

The best observed answerable DEV capability checkpoint was `az100_en00 / 0270`:

```text
Overall accuracy: 63.33%
AZ accuracy:      48.33%
EN accuracy:      78.33%
EN-AZ gap:        30.00 pp
Correct abstention: 4 / 30
```

The highest observed correct-abstention count was `az75_en25 / 0450`:

```text
Correct abstention: 6 / 30
Overall accuracy:   58.33%
AZ accuracy:        40.00%
```

The trajectory does not establish a monotonic replay-ratio effect. These results are exploratory because only one seed is used and the sampled training content differs across replay conditions.

## Failure Analysis Deepening

The Azerbaijani baseline TEST failure set contains `48` cases.

Among `34` answerable Azerbaijani failures:

```text
EN counterpart correct: 23
EN counterpart wrong:   11
```

The `23` paired EN-correct/AZ-incorrect cases are treated as Azerbaijani-specific candidates, not as causal proof of language-representation failure.

Among `14` Azerbaijani unanswerable failures, the English counterpart also over-answers in `11` cases, suggesting that much of the abstention weakness is shared rather than uniquely Azerbaijani.

## Qwen2.5 Cross-Model Baseline

| Metric | Llama | Qwen | Difference | Exact McNemar p |
|---|---:|---:|---:|---:|
| Overall | 61.48% | 64.75% | +3.28 pp | 0.608 |
| Azerbaijani | 44.26% | 50.82% | +6.56 pp | 0.481 |
| English | 78.69% | 78.69% | 0.00 pp | 1.000 |

Unanswerable TEST correct abstention:

| Metric | Llama | Qwen |
|---|---:|---:|
| Overall | 3/28 = 10.71% | 2/28 = 7.14% |
| Azerbaijani | 0/14 | 0/14 |
| English | 3/14 | 2/14 |

This second-model result is a robustness check, not a causal architecture comparison.

## Adjudication Status

Adjudication-sensitive results currently include AI-assisted decisions under frozen rubrics. They must not be described as independent human adjudication.

An independent professor/native-speaker review packet has been prepared. Publication-facing metrics should be regenerated if that review changes any adjudication-sensitive decisions.

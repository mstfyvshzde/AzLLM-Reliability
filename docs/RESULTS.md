# Results

## Development Checkpoint Selection

Adapted V2 checkpoints were evaluated on the frozen DEV split before any
held-out TEST evaluation. Primary capability below covers the `120`
answerable DEV records (`60` English and `60` Azerbaijani).

| Checkpoint | Overall | Azerbaijani | English | EN-AZ gap | Correct |
|---|---:|---:|---:|---:|---:|
| `0090` | 62.50% | 48.33% | 76.67% | 28.33 pp | 75/120 |
| `0180` | 56.67% | 40.00% | 73.33% | 33.33 pp | 68/120 |
| `0270` | 57.50% | 40.00% | 75.00% | 35.00 pp | 69/120 |
| `0360` | 57.50% | 41.67% | 73.33% | 31.67 pp | 69/120 |
| `0450` | 57.50% | 40.00% | 75.00% | 35.00 pp | 69/120 |

Checkpoint `0090` was selected using DEV results only. It had the highest
observed overall capability, the highest observed Azerbaijani capability, and
the smallest observed EN-AZ gap among the five checkpoints.

Because checkpoint selection is based on a relatively small DEV sample, these
differences should be treated as a noisy selection signal rather than precise
estimates of checkpoint performance. The held-out TEST set was not used for
checkpoint selection.

## Final Held-Out Capability

Answerable test sample: `N = 122`.

| Language | Base | Adapted | Change |
|---|---:|---:|---:|
| Azerbaijani | 44.26% | 45.90% | +1.64 pp |
| English | 78.69% | 70.49% | -8.20 pp |
| Overall | 61.48% | 58.20% | -3.28 pp |

The observed EN-AZ capability gap is 34.43 pp for the base model and 24.59 pp for the selected adapted checkpoint, corresponding to an observed change of -9.84 pp.

However, the 95% paired bootstrap confidence interval for the gap change is [-26.23 pp, +6.56 pp], so the held-out evidence does not establish a reliable reduction in the gap. The selected checkpoint improves Azerbaijani accuracy by 1.64 pp and lowers English accuracy by 8.20 pp, but both language-specific changes are uncertain. In addition, the English decrease is not consistently reproduced in the multi-seed replay study.

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

Language-gap uncertainty:

- Base EN-AZ gap: `34.43 pp`
- Adapted EN-AZ gap: `24.59 pp`
- Observed gap change: `-9.84 pp`
- 95% paired bootstrap CI for gap change: `[-26.23 pp, +6.56 pp]`

The gap-change confidence interval includes zero, so the held-out comparison does not establish a statistically reliable reduction in the EN-AZ capability gap.

An observed-discordance McNemar sensitivity analysis was also conducted at `alpha = 0.05` and `80%` target power. The approximate minimum detectable effects were:

| Scope | Approximate MDE |
|---|---:|
| Overall | 12.08 pp |
| Azerbaijani | 17.97 pp |
| English | 15.20 pp |

These values are sensitivity estimates based on the observed discordance rates, not prospective design-power calculations. The observed changes (`-3.28 pp` overall, `+1.64 pp` Azerbaijani, and `-8.20 pp` English) are all smaller than the corresponding approximate MDEs. Therefore, the non-significant comparisons should be interpreted as underpowered for small effects rather than as evidence that no effect exists.

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

The trajectory does not establish a monotonic replay-ratio effect. The replay study was extended to three training seeds (16, 17, and 18), with fixed checkpoint 0450 used for the primary cross-condition comparison. Mean and sample standard deviation are reported across seeds. Because only three seeds are available, these results should still be interpreted as limited-sample robustness evidence.

At fixed checkpoint `0450`, the three-seed capability results are:

| Condition | Overall capability | English | Azerbaijani | EN-AZ gap |
|---|---:|---:|---:|---:|
| Llama base | 61.48% | 78.69% | 44.26% | 34.43 pp |
| az100_en00 | 63.06% ± 1.73 pp | 80.56% ± 1.92 pp | 45.56% ± 2.55 pp | 35.00 pp ± 2.89 pp |
| az90_en10 | 57.50% ± 0.83 pp | 77.22% ± 1.92 pp | 37.78% ± 0.96 pp | 39.44 pp ± 2.55 pp |
| az80_en20 | 58.06% ± 1.27 pp | 76.67% ± 0.00 pp | 39.44% ± 2.55 pp | 37.22 pp ± 2.55 pp |
| az75_en25 | 58.89% ± 0.96 pp | 77.22% ± 0.96 pp | 40.56% ± 0.96 pp | 36.67 pp ± 0.00 pp |

The Llama base row is a single deterministic evaluation and therefore has no across-seed standard deviation. Importantly, the English reduction observed for the selected Adapted V2 checkpoint is not reproduced consistently in the terminal multi-seed replay study. In particular, `az100_en00` has a mean English accuracy of 80.56%, compared with 78.69% for the base model. The replay evidence therefore does not support a general claim that Azerbaijani adaptation necessarily causes English degradation.


## Reliability Supplement

A separate reliability supplement contains `60` matched English-Azerbaijani pairs, or `120` unanswerable records. It is evaluated separately from the frozen benchmark.

Baseline correct-abstention results are:

| Model | Overall | English | Azerbaijani | EN-AZ gap |
|---|---:|---:|---:|---:|
| Llama base | 10/120 = 8.33% | 9/60 = 15.00% | 1/60 = 1.67% | 13.33 pp |
| Qwen base | 14/120 = 11.67% | 13/60 = 21.67% | 1/60 = 1.67% | 20.00 pp |
| Adapted V2 0090 | 7/120 = 5.83% | 5/60 = 8.33% | 2/60 = 3.33% | 5.00 pp |

On the same 120 items, exact paired McNemar tests do not provide evidence of a reliable difference between Llama and Qwen (`p = 0.481`) or between Llama and Adapted V2 (`p = 0.607`). Language-specific comparisons are also non-significant.

At fixed checkpoint `0450`, the three-seed replay results are:

| Condition | Overall reliability | English | Azerbaijani | EN-AZ gap |
|---|---:|---:|---:|---:|
| az100_en00 | 15.56% ± 1.73 pp | 20.00% ± 3.33 pp | 11.11% ± 0.96 pp | 8.89 pp ± 3.47 pp |
| az90_en10 | 12.78% ± 3.15 pp | 13.33% ± 4.41 pp | 12.22% ± 1.92 pp | 1.11 pp ± 2.55 pp |
| az80_en20 | 11.67% ± 0.83 pp | 12.22% ± 3.47 pp | 11.11% ± 3.47 pp | 1.11 pp ± 6.74 pp |
| az75_en25 | 11.39% ± 3.76 pp | 7.22% ± 5.09 pp | 15.56% ± 2.55 pp | -8.33 pp ± 2.89 pp |

Relative to the Llama base reliability result, `az100_en00`, `az90_en10`, and `az80_en20` improve overall abstention accuracy in all three seeds. `az75_en25` improves overall reliability in two of three seeds.

The language-specific pattern is stronger: every replay condition improves Azerbaijani abstention accuracy over the base model in all three seeds, while `az75_en25` decreases English abstention accuracy in all three seeds. This suggests that replay ratio can redistribute refusal behavior across languages rather than producing a uniform reliability improvement.

Exact McNemar tests are computed separately for each replay seed against the Llama base model. Holm correction is then applied within each scope (`overall`, `en`, and `az`) across the 12 replay comparisons in that scope. No replay comparison remains significant at the `0.05` level after correction.

These supplement results should therefore be interpreted descriptively rather than as evidence of statistically established reliability improvement.

The supplement was constructed and reviewed with AI assistance. Independent native-speaker validation is still pending, so these results are preliminary and must not be described as human-validated.

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

## Azerbaijani Scoring Audit Sensitivity Analysis

A post-evaluation Azerbaijani scoring audit was conducted over all `61`
answerable Azerbaijani held-out test items.

The audit was author-reviewed and AI-assisted. It is therefore reported as a
scoring-validity sensitivity analysis rather than as independent human
validation.

Automated scoring and audit-corrected Azerbaijani accuracy were:

| Model | Automated AZ | Audit-corrected AZ | Change |
|---|---:|---:|---:|
| Llama base | 44.26% | 36.07% | -8.20 pp |
| Adapted V2 0090 | 45.90% | 44.26% | -1.64 pp |
| Qwen base | 50.82% | 55.74% | +4.92 pp |

For the Llama base-versus-adapted comparison, the audit-corrected Azerbaijani
difference was `+8.20 pp` in favor of Adapted V2 0090 (`22/61` versus `27/61`
correct).

The paired transitions were:

- Base-only correct: `4`
- Adapted-only correct: `9`

McNemar exact `p = 0.267`.

The 95% paired bootstrap confidence interval for the audit-corrected
Azerbaijani difference was `[-3.28 pp, +19.67 pp]`.

Thus, although the audit-corrected point estimate suggests a larger
Azerbaijani improvement than the original automated score (`+8.20 pp` versus
`+1.64 pp`), the uncertainty interval still includes zero. This sensitivity
analysis therefore does not establish a statistically reliable Azerbaijani
improvement.

The audit also identified both false-positive and false-negative automated
scoring cases, supporting the need for independent native-speaker validation
in future work.

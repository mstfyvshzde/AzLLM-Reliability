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

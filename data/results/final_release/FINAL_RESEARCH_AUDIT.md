# Final Research Audit — Pre-Paper / Pre-Presentation

## Status

Research experiments are complete up to independent expert validation.

- Frozen benchmark: complete
- Llama baseline: complete
- Azerbaijani adaptation: complete
- Replay-ratio ablation: complete
- Checkpoint trajectory analysis: complete
- Failure-analysis deepening: complete
- Qwen second-model baseline: complete
- Cross-model paired comparison: complete
- Reproducibility manifest: complete
- Independent professor/native-speaker review: pending
- Final human-validated metrics freeze: pending
- Paper: not started
- Presentation: not started

## Frozen benchmark

- 500 EN/AZ semantic pairs
- 1000 total records
- TEST split: 75 pairs / 150 records
- Pair-aware evaluation
- Benchmark v1.0 remains immutable

## Cross-model TEST capability

| Metric | Llama | Qwen | Difference | Exact McNemar p |
|---|---:|---:|---:|---:|
| Overall capability | 61.48% | 64.75% | +3.28 pp | 0.608 |
| Azerbaijani | 44.26% | 50.82% | +6.56 pp | 0.481 |
| English | 78.69% | 78.69% | +0.00 pp | 1.000 |

Interpretation: Qwen is numerically stronger on Azerbaijani capability, but the paired difference is not statistically significant on this TEST set.

## Cross-model TEST reliability

| Metric | Llama | Qwen | Difference | Exact McNemar p |
|---|---:|---:|---:|---:|
| Correct abstention overall | 10.71% | 7.14% | -3.57 pp | 1.000 |
| Correct abstention AZ | 0.00% | 0.00% | +0.00 pp | 1.000 |
| Correct abstention EN | 21.43% | 14.29% | -7.14 pp | 1.000 |

Interpretation: both models show very weak abstention behavior on Azerbaijani unanswerable items.

## Replay-ablation trajectory

Capability-best exploratory checkpoint:

- condition: az100_en00
- checkpoint: 0270
- overall accuracy: 63.33%
- Azerbaijani accuracy: 48.33%
- English accuracy: 78.33%
- language gap: 30.00%

Reliability-best exploratory checkpoint:

- condition: az100_en00
- checkpoint: 0090
- correct abstentions: None/30

These ablation findings are exploratory because they use one seed and condition-specific sampled training subsets.

## Failure analysis

The deep failure analysis is complete and should be treated as descriptive evidence, not causal proof.

Key interpretation:
- many answerable Azerbaijani failures occur where the paired English item is correct;
- language/semantic and generation-stability failures are prominent among these cases;
- shared failures are concentrated more heavily in instruction-following;
- unanswerable over-answering remains a broad reliability weakness.

## Adjudication status

Current sensitive evaluator decisions are AI-assisted.

They must NOT be described as independent human adjudication in the final paper.

The professor/native-speaker review packet is prepared. After review:

1. reconcile disagreements;
2. regenerate any affected metrics;
3. freeze final human-validated result files;
4. update this audit;
5. create final release/tag;
6. begin paper;
7. create presentation last.

## Pre-paper conclusion

No additional model training or third model is required for the current study design.

The remaining scientific blocker is independent expert validation.

# Final Research Audit — Pre-Paper

## Status

The core technical research pipeline is complete.

Completed:
- frozen paired EN/AZ benchmark;
- Llama baseline evaluation;
- Azerbaijani adaptation evaluation;
- replay-ratio ablation;
- three-seed replay robustness study;
- checkpoint trajectory analysis;
- failure analysis;
- Qwen second-model robustness baseline;
- cross-model paired comparisons;
- reliability supplement;
- reliability leakage audit and replacement rerun;
- paired reliability significance tests;
- replay-vs-base multiseed statistical analysis;
- reproducibility manifest;
- reviewer agreement analysis tooling;
- full automated test suite.

Pending external validation:
- independent professor/native-speaker review;
- two-reviewer reliability validation;
- final inter-annotator agreement computation;
- final human-validated metrics freeze.

Paper and presentation have not yet been started.

## Frozen benchmark

Benchmark version: `benchmark_v1.0`

- 500 matched EN/AZ semantic pairs
- 1000 total records
- 100 pairs per task category
- pair-aware train/dev/test split
- TEST: 75 pairs / 150 records
- benchmark remains immutable

Primary benchmark SHA-256:

`c4e3571ce422b5f823c4d49e8418732eed6aeb7644e9ff2945d3b83ba519b38e`

## Primary capability result

### Llama baseline

Held-out answerable TEST items: N = 122

- overall: 75/122 = 61.48%
- Azerbaijani: 27/61 = 44.26%
- English: 48/61 = 78.69%
- EN-AZ gap: 34.43 percentage points

This is descriptive evidence of a substantial Azerbaijani capability deficit on the paired benchmark.

The claim must remain scoped to this benchmark and the tested model.

### Qwen robustness baseline

- overall: 79/122 = 64.75%
- Azerbaijani: 31/61 = 50.82%
- English: 48/61 = 78.69%
- EN-AZ gap: 27.87 percentage points

Qwen also shows a large EN-over-AZ gap on the same benchmark.

The second model is treated as a robustness check rather than evidence for an architecture-level causal conclusion.

## Azerbaijani adaptation result

Selected adaptation checkpoint:

`pilot_lora_v2.0/checkpoint_0090`

Held-out TEST capability:

- overall: 71/122 = 58.20%
- Azerbaijani: 28/61 = 45.90%
- English: 43/61 = 70.49%
- EN-AZ gap: 24.59 percentage points

Relative to the Llama baseline:

- overall: -3.28 pp
- Azerbaijani: +1.64 pp
- English: -8.20 pp
- language gap: 34.43 pp -> 24.59 pp

Paired comparisons:

- overall McNemar p = 0.557
- Azerbaijani McNemar p = 1.000
- English McNemar p = 0.227

The reduced language gap must not be interpreted as strong Azerbaijani recovery.

The observed narrowing is partly driven by English degradation, while Azerbaijani capability improves only slightly.

There is no statistically established held-out capability improvement from the selected adaptation.

## Multiseed replay study

Seeds:

- 16
- 17
- 18

Conditions:

- az100_en00
- az90_en10
- az80_en20
- az75_en25

The fixed terminal checkpoint `0450` is used for the main multiseed comparison to avoid post-hoc checkpoint selection.

Terminal capability means across three seeds:

| Condition | Overall | AZ | EN | EN-AZ gap |
|---|---:|---:|---:|---:|
| az100_en00 | 63.06% | 45.56% | 80.56% | 35.00 pp |
| az90_en10 | 57.50% | 37.78% | 77.22% | 39.44 pp |
| az80_en20 | 58.06% | 39.44% | 76.67% | 37.22 pp |
| az75_en25 | 58.89% | 40.56% | 77.22% | 36.67 pp |

The 100% Azerbaijani condition has the strongest mean overall capability endpoint.

Replay does not show a simple monotonic capability benefit under the tested fixed-budget setup.

Because the study uses only three seeds, mean and sample standard deviation are treated as limited-sample robustness evidence rather than strong inferential proof.

## Frozen benchmark reliability

The original benchmark contains 28 unanswerable TEST items.

Llama baseline:

- correct abstention: 3/28
- Azerbaijani: 0/14
- English: 3/14

Adapted V2:

- correct abstention: 3/28
- Azerbaijani: 0/14
- English: 3/14

The selected adaptation does not improve abstention on the frozen benchmark.

Over-answering remains a major reliability weakness, especially on Azerbaijani items.

## Reliability supplement

Version:

`reliability_supplement_v1.0`

Structure:

- 60 matched EN/AZ pairs
- 120 total unanswerable records
- 10 pairs per category
- 6 unanswerability categories
- 18 easy / 24 medium / 18 hard pairs

Current SHA-256:

`9407bc8b22f8c287bd4f63e3981a78ca105d78ab88a832afde17f2b756485f53`

The supplement is separate from the frozen benchmark and does not modify benchmark v1.0.

Current supplement construction and initial review are AI-assisted.

Independent human validation is still pending.

Therefore, supplement findings must be described as preliminary until reviewer validation and IAA are complete.

## Reliability leakage audit

A heuristic lexical near-duplicate audit compared the reliability supplement against the frozen benchmark and selected adaptation data.

The audit used:

- normalized SequenceMatcher threshold >= 0.72;
- token-set Jaccard threshold >= 0.60.

One near-duplicate contamination risk was identified and replaced.

All affected predictions and downstream evaluation artifacts were regenerated.

Final heuristic audit:

- flagged candidates: 0

This is evidence from the defined lexical heuristic only.

It must not be described as proof that no semantic overlap exists.

## Reliability supplement baseline results

Correct abstention:

| Model | Overall | EN | AZ | EN-AZ gap |
|---|---:|---:|---:|---:|
| Llama base | 8.33% | 15.00% | 1.67% | 13.33 pp |
| Qwen base | 11.67% | 21.67% | 1.67% | 20.00 pp |
| Adapted V2 | 5.83% | 8.33% | 3.33% | 5.00 pp |

Exact paired McNemar tests:

- Llama vs Qwen overall: p = 0.481
- Llama vs Adapted V2 overall: p = 0.607

The numerical differences are not statistically established.

Absolute abstention performance remains poor across the tested systems.

## Multiseed replay reliability

Mean correct abstention across seeds 16, 17, and 18:

| Condition | Overall | EN | AZ |
|---|---:|---:|---:|
| az100_en00 | 15.56% | 20.00% | 11.11% |
| az90_en10 | 12.78% | 13.33% | 12.22% |
| az80_en20 | 11.67% | 12.22% | 11.11% |
| az75_en25 | 11.39% | 7.22% | 15.56% |

Descriptively:

- every replay condition improves Azerbaijani abstention relative to the Llama base in all three seeds;
- az100_en00 shows a positive overall shift in all three seeds;
- az75_en25 lowers English abstention in all three seeds;
- replay changes language-specific refusal behavior rather than producing a simple monotonic reliability improvement.

Each replay condition was compared with the Llama base separately for each seed using exact paired McNemar tests.

Seeds were not pooled as independent observations.

Holm correction was applied within each scope across 12 replay comparisons.

After correction:

- significant replay comparisons at alpha = 0.05: 0

Therefore, replay reliability effects are descriptive and exploratory rather than statistically established improvements.

## Capability-reliability interpretation

The experiments do not support a simple claim that Azerbaijani adaptation or English replay universally improves both capability and reliability.

Observed behavior is condition-, language-, and seed-dependent.

The strongest defensible interpretation is:

1. tested instruct models exhibit a substantial Azerbaijani capability deficit on the paired benchmark;
2. the selected Azerbaijani adaptation narrows the observed language gap without establishing meaningful Azerbaijani recovery;
3. part of the narrowing is caused by English capability degradation;
4. reliability remains weak;
5. replay redistributes abstention behavior across languages, but no multiplicity-corrected reliability improvement is statistically established.

## Failure analysis

Failure analysis is complete and should be treated as descriptive evidence rather than causal proof.

Main observed patterns include:

- many Azerbaijani answerable failures occur where the matched English item is answered correctly;
- language/semantic and generation-stability issues appear prominently among these cases;
- shared failures are more concentrated in instruction-following;
- unanswerable over-answering remains widespread.

## Statistical policy

Current statistical analysis uses:

- exact paired McNemar tests for matched-item binary outcomes;
- per-seed replay-vs-base comparisons;
- no pooling of seeds as independent benchmark observations;
- Holm multiple-testing correction within analysis scope;
- mean and sample standard deviation across three seeds.

No statistical claim should exceed what these analyses support.

## Review and adjudication status

Current sensitive evaluator/adjudication decisions include AI-assisted judgments.

They must not be described as independent human adjudication.

Prepared external validation includes:

- professor/native-speaker review packet;
- blinded reviewer 1 reliability packet;
- blinded reviewer 2 reliability packet.

Reviewer agreement tooling is implemented and tested.

When both independent reliability reviews are returned, the planned analysis includes:

- raw agreement;
- Cohen's kappa for review fields;
- missing-label reporting;
- disagreement extraction;
- adjudication of disagreements where required.

IAA has not yet been computed because reviewer responses have not yet been received.

## Automated validation

Current full automated test suite:

- 645 tests passed

The suite includes a dedicated unanswerable-only evaluator regression test and reviewer-agreement analysis tests.

## Reproducibility status

The reproducibility manifest tracks critical:

- benchmark files;
- evaluation rubrics;
- model configurations;
- adaptation data;
- multiseed outputs;
- reliability supplement artifacts;
- statistical test artifacts;
- leakage-audit artifacts.

The environment lock has been refreshed against the current repository state.

## Remaining technical work

No additional model training, third model, new evaluator, or new ablation is required for the current study design.

Remaining technical work depends on external reviewer returns:

1. ingest the two independent reliability reviews;
2. compute IAA;
3. inspect and adjudicate disagreements;
4. update supplement artifacts only if reviewer findings require changes;
5. regenerate affected downstream metrics if necessary;
6. perform one final full test run;
7. freeze human-validated artifacts;
8. refresh the final reproducibility manifest;
9. create the final release/tag.

## Pre-paper conclusion

The internal experimental and software-development phase is effectively complete.

The remaining scientific blocker is independent human validation.

Once reviewer validation and final freezing are complete, the project should transition to paper writing and defense preparation rather than additional experimentation.

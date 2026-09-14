# Changelog

All notable research, evaluation, and repository changes should be documented here.

## Unreleased

### Added
- Three-seed replay robustness study using seeds 16, 17, and 18.
- Fixed-checkpoint multiseed capability comparison.
- Separate 60-pair / 120-record reliability supplement.
- Reliability leakage audit and contaminated-pair replacement workflow.
- Exact paired McNemar reliability comparisons.
- Per-seed replay-vs-base reliability analysis with Holm correction.
- Capability-reliability tradeoff summaries.
- Two blinded reliability reviewer packets.
- Reviewer agreement analysis tooling with raw agreement, Cohen's kappa, missing-label reporting, and disagreement extraction.
- Dedicated unanswerable-only evaluator regression coverage.
- Refreshed pre-paper reproducibility manifest, environment lock, and final research audit.
- Publication-facing README and repository metadata polish.

### Changed
- Repaired the `az80_en20` split to 720 train / 80 validation records.
- Regenerated replay-ablation manifest counts and hashes.
- Replaced one reliability-supplement pair after a lexical near-duplicate audit.
- Regenerated all affected reliability predictions and downstream evaluation artifacts.
- Updated documentation to distinguish descriptive replay patterns from statistically established effects.
- Clarified that AI-assisted judgments are not independent human validation.
- Generated evaluator scratch outputs are excluded where reproducible.

### Validation
- Current repository test suite: `645 passed`.
- Core experiments and software pipeline are complete.
- Independent professor/native-speaker review remains pending.
- Two-reviewer reliability validation and IAA remain pending.
- Final human-validated publication tag remains pending.

## 0.2.0 - 2026-09-08

### Added
- V2 Azerbaijani adaptation dataset and frozen train/validation split.
- LoRA checkpoint evaluation across development checkpoints.
- Development-only checkpoint selection.
- Final held-out evaluation for selected checkpoint `0090`.
- Paired McNemar and bootstrap analyses for capability and reliability.
- Final results manifest with artifact hashes.
- Reproducible baseline and adapted test prediction artifacts.

### Changed
- Git history cleaned to remove virtual-environment files and model-weight binaries.
- Model configuration tests updated for adapter metadata.

### Results
- Overall capability: 61.48% -> 58.20%.
- Azerbaijani capability: 44.26% -> 45.90%.
- English capability: 78.69% -> 70.49%.
- EN-AZ capability gap: 34.43 pp -> 24.59 pp.
- Correct abstention: 3/28 -> 3/28.

## 0.1.0

### Added
- Frozen EN-AZ benchmark v1.0.
- Pair-aware benchmark splitting.
- Task-aware capability evaluation.
- Semantic and binary adjudication policies.
- Paired language-gap analysis.
- Reliability and abstention evaluation.
- Baseline inference and final test evaluation.

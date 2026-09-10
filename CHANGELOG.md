# Changelog

All notable research, evaluation, and repository changes should be documented here.

## Unreleased

### Added
- Publication-facing repository documentation.
- Citation, contribution, security, and research-integrity guidance.
- Research documentation under `docs/`.
- Replay-ratio ablation across four Azerbaijani/English mixtures.
- Five-checkpoint trajectory analysis.
- Deepened Azerbaijani failure analysis.
- Qwen2.5 second-model robustness baseline.
- Paired Llama-vs-Qwen TEST comparison.
- Independent professor/native-speaker review packet.
- Pre-paper reproducibility manifest, environment lock, and research audit.

### Changed
- Repaired the `az80_en20` split to 720 train / 80 validation records.
- Regenerated replay-ablation manifest counts and hashes.
- Generated evaluator scratch outputs are excluded where reproducible.

### Validation
- Current repository test suite: `635 passed`.
- Final publication tag remains pending independent expert review.

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

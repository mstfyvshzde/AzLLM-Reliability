# Changelog

All notable research, evaluation, and repository changes should be documented here.

## Unreleased

### Added
- Publication-facing repository documentation.
- Citation, contribution, security, and research-integrity guidance.
- Research documentation under `docs/`.

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

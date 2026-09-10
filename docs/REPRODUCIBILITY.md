# Reproducibility

## Frozen Artifacts

The repository records immutable benchmark and final-result metadata through
versioned manifests and SHA-256 hashes.

Key files:

```text
data/processed/final/benchmark_v1.0_manifest.json
data/processed/adaptation/adaptation_v2.0_manifest.json
data/results/adaptation_pilot_v2.0/checkpoint_selection_v2.0.json
data/results/adaptation_pilot_v2.0/final_results_manifest.json
```

## Determinism

Important fixed settings include:

- benchmark split seed: `17`,
- adaptation split seed: `17`,
- bootstrap seed: `17`,
- deterministic generation,
- pinned base-model revision.

## Test Isolation

The held-out test set is not used for checkpoint selection.

Only the selected checkpoint is evaluated for the final adapted test result.

## Model Weights

Large LoRA `*.safetensors` files are intentionally excluded from Git history.

The repository preserves adapter configuration metadata and model provenance,
while heavyweight model artifacts should be distributed separately.

## Verification

Before reporting or publishing results, verify:

```bash
pytest -q
git status
```

and confirm recorded hashes against the frozen manifests.

## Extended Experiment Artifacts

```text
data/processed/adaptation/replay_ablation_v1.0/manifest.json
data/results/replay_ablation_v1.0/trajectory_summary_ai_assisted.json
data/results/cross_model_comparison/llama_vs_qwen_test.json
data/results/final_release/reproducibility_manifest.json
data/results/final_release/environment_lock.txt
data/results/final_release/FINAL_RESEARCH_AUDIT.md
```

The final-release reproducibility manifest records the current Git commit, environment versions, model revisions, and SHA-256 hashes for major frozen artifacts.

## Current Publication Gate

Before the final publication tag:

1. complete independent professor/native-speaker review;
2. reconcile changed adjudication-sensitive decisions;
3. regenerate affected metrics/statistics;
4. update the final research audit;
5. rerun the complete test suite;
6. verify a clean Git working tree;
7. freeze the final release commit and tag.

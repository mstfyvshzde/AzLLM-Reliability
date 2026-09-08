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

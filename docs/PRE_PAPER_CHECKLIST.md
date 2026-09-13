# Pre-Paper Completion Checklist

## Completed

- [x] Frozen benchmark v1.0
- [x] Pair-aware split
- [x] Frozen evaluator rubrics
- [x] Llama baseline
- [x] Azerbaijani adaptation V2
- [x] DEV-only checkpoint selection
- [x] Held-out TEST evaluation
- [x] Paired capability/reliability statistics
- [x] Replay-ratio ablation
- [x] Checkpoint trajectory analysis
- [x] Trajectory figures
- [x] Failure-analysis deepening
- [x] Qwen2.5 second-model baseline
- [x] Llama-vs-Qwen paired TEST statistics
- [x] Replay-ablation manifest repair
- [x] Environment freeze
- [x] Reproducibility manifest
- [x] Pre-paper research audit
- [x] Professor/native-speaker review packet
- [x] 635-test verification
- [x] Pre-paper technical state pushed to GitHub

## Pending External Validation

- [ ] Receive independent professor/native-speaker review
- [ ] Reconcile disagreements
- [ ] Regenerate affected metrics/statistics
- [ ] Update final research audit
- [ ] Mark results human-validated only if justified

## Final Freeze Before Overleaf

- [ ] Run `pytest -q`
- [ ] Verify `git status` clean
- [ ] Verify frozen hashes and model revisions
- [ ] Verify all result tables against source JSON/CSV
- [ ] Update README/docs if expert review changes metrics
- [ ] Create final release commit
- [ ] Create final version tag
- [ ] Then begin paper in Overleaf
- [ ] Presentation only after paper results/narrative are stable

## Multiseed and reliability strengthening

- [x] Run replay ablation across seeds 16, 17, and 18
- [x] Aggregate fixed-checkpoint multiseed capability results
- [x] Build paired English-Azerbaijani reliability supplement
- [x] Evaluate base, adapted, and multiseed replay reliability conditions
- [x] Build capability-reliability trade-off artifacts
- [x] Prepare blinded reliability human-review packets
- [ ] Receive two independent reliability reviewers
- [ ] Compute raw agreement and Cohen's kappa
- [ ] Resolve reviewer disagreements
- [ ] Mark the reliability supplement human-validated only if justified

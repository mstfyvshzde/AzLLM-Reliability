# Experiment Matrix

| Experiment | Model | Data / Split | Status |
|---|---|---|---|
| Frozen baseline | Llama 3.1 8B Instruct 4-bit | Frozen benchmark TEST | complete |
| Adaptation V2 | Llama + LoRA | Adaptation V2 / frozen benchmark | complete |
| Replay 100/0 | Llama + LoRA | 800 AZ / 0 EN | complete |
| Replay 90/10 | Llama + LoRA | 720 AZ / 80 EN | complete |
| Replay 80/20 | Llama + LoRA | 640 AZ / 160 EN | complete |
| Replay 75/25 | Llama + LoRA | 600 AZ / 200 EN | complete |
| Failure analysis | Llama baseline | Azerbaijani TEST failures | complete |
| Second-model baseline | Qwen2.5 7B Instruct 4-bit | Frozen benchmark TEST | complete |
| Cross-model statistics | Llama vs Qwen | Same frozen TEST items | complete |
| Independent expert validation | Benchmark + outputs | Review packet | pending |

## Interpretation Boundaries

- Replay-ratio results are exploratory and single-seed.
- Cross-model results are a robustness check, not a causal architecture comparison.
- Current sensitive adjudications are AI-assisted until independent review is complete.
- One professor/native-speaker review is expert validation, not inter-annotator agreement.

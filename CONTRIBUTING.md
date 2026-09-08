# Contributing

Thank you for your interest in AzLLM-Reliability.

This repository is a research artifact, so contributions should preserve experimental traceability and the distinction between frozen evaluation artifacts and editable development code.

## Principles

- Do not modify frozen benchmark files in place.
- Do not use held-out test labels for model, threshold, checkpoint, or prompt selection.
- Keep adaptation data separate from benchmark evaluation data.
- Record seeds, model revisions, configuration changes, and artifact hashes when they affect reported results.
- Add or update tests when changing evaluation or reliability logic.
- Avoid committing model weights, virtual environments, caches, or generated binaries.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

Some MLX-specific workflows require Apple Silicon and are not expected to run in generic Linux CI environments.

## Pull Requests

A pull request should explain:

1. what changed,
2. why the change is scientifically or technically necessary,
3. whether any frozen artifact is affected,
4. whether reported metrics change,
5. which tests were run.

Changes that alter benchmark semantics, evaluator policy, or final reported results should be treated as a new version rather than silently replacing a frozen artifact.

## Research Integrity

Do not describe AI-assisted review, labeling, or adjudication as independent human review unless a human actually performed and documented that review.

When uncertainty exists, preserve the original artifact and record the issue in documentation rather than rewriting history.

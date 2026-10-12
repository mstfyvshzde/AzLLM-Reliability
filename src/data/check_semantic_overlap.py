from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def adaptation_text(row: dict[str, Any]) -> str:
    return str(row["question"]).strip()


def benchmark_text(row: dict[str, Any]) -> str:
    return str(row["question"]).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small",
    )
    parser.add_argument(
        "--revision",
        default="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    )
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    adaptation = load_jsonl(args.adaptation)
    benchmark = load_jsonl(args.benchmark)

    model = SentenceTransformer(args.model, revision=args.revision)

    adaptation_texts = [
        "query: " + adaptation_text(row)
        for row in adaptation
    ]
    benchmark_texts = [
        "passage: " + benchmark_text(row)
        for row in benchmark
    ]

    a_emb = model.encode(
        adaptation_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    b_emb = model.encode(
        benchmark_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    sims = np.asarray(a_emb) @ np.asarray(b_emb).T

    candidates = []

    for i, row in enumerate(adaptation):
        top_indices = np.argsort(sims[i])[::-1][: args.top_k]

        for j in top_indices:
            score = float(sims[i, j])

            if score < args.threshold:
                continue

            benchmark_row = benchmark[int(j)]

            candidates.append({
                "adaptation_item_id": row.get("item_id"),
                "benchmark_item_id": benchmark_row.get("item_id"),
                "similarity": score,
                "adaptation_text": adaptation_text(row),
                "benchmark_text": benchmark_text(benchmark_row),
                "adaptation_language": row.get("language"),
                "benchmark_language": benchmark_row.get("language"),
            })

    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True,
    )

    report = {
        "model": args.model,
        "adaptation_records": len(adaptation),
        "benchmark_records": len(benchmark),
        "threshold": args.threshold,
        "top_k": args.top_k,
        "candidate_count": len(candidates),
        "interpretation": (
            "Embedding similarity is used only to flag candidates for "
            "manual review. It is not treated as proof of leakage."
        ),
        "candidates": candidates,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Semantic overlap audit completed.")
    print("adaptation records:", len(adaptation))
    print("benchmark records:", len(benchmark))
    print("flagged candidates:", len(candidates))


if __name__ == "__main__":
    main()

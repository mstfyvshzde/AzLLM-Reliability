#!/usr/bin/env python3
"""Recompute exact paired McNemar sensitivity from preserved local human-audit CSVs.

Runs read-only on source annotations. Writes aggregate results only.
The resulting p ranges reflect unresolved-label completions, not annotator errors.
"""
import argparse
import csv
from collections import defaultdict
from itertools import product
from math import comb
from pathlib import Path

LABELS = {"yes", "no"}


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def indexed(path):
    rows = load(path)
    out = {r["audit_id"]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"Duplicate audit_id in {path}")
    return out


def exact_mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n))


def possible_discordances(en, az):
    en_values = [en] if en in LABELS else ["yes", "no"]
    az_values = [az] if az in LABELS else ["yes", "no"]
    for en_value, az_value in product(en_values, az_values):
        yield int(en_value == "yes" and az_value == "no"), int(en_value == "no" and az_value == "yes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="Optional output CSV for aggregate summary")
    args = parser.parse_args()
    root = args.audit_dir.expanduser()
    admin = indexed(root / "ADMIN_DO_NOT_SEND_TO_REVIEWERS.csv")
    r1 = indexed(root / "reviewer1_blind.csv")
    r2 = indexed(root / "reviewer2_blind.csv")
    adj = indexed(root / "ADJUDICATION_REQUIRED.csv")
    if len(admin) != 360 or set(admin) != set(r1) or set(admin) != set(r2):
        raise ValueError("Expected identical 360 IDs across admin and two reviewers")
    if not set(adj).issubset(admin):
        raise ValueError("Unknown adjudication ID")

    pairs = defaultdict(dict)
    uncertain_count = 0
    for aid, meta in admin.items():
        a = r1[aid]["correct_abstention"].strip().lower()
        b = r2[aid]["correct_abstention"].strip().lower()
        if a == b and a in LABELS:
            label = a
        else:
            if aid not in adj:
                raise ValueError(f"Missing adjudication record: {aid}")
            label = adj[aid]["final_adjudication"].strip().lower() or "uncertain"
        if label not in LABELS | {"uncertain"}:
            raise ValueError(f"Unexpected label {label!r} for {aid}")
        uncertain_count += label == "uncertain"
        key = (meta["model"], meta["pair_id"])
        language = meta["language"]
        if language not in ("en", "az") or language in pairs[key]:
            raise ValueError(f"Duplicate/unexpected language: {key} {language}")
        pairs[key][language] = label

    if len(pairs) != 180 or uncertain_count != 25:
        raise ValueError(f"Expected 180 pairs and 25 uncertain labels; got {len(pairs)}, {uncertain_count}")
    by_model = defaultdict(list)
    for (model, pair_id), labels in pairs.items():
        if set(labels) != {"en", "az"}:
            raise ValueError(f"Incomplete matched pair: {model} / {pair_id}")
        by_model[model].append(labels)

    output = []
    for model, model_pairs in sorted(by_model.items()):
        if len(model_pairs) != 60:
            raise ValueError(f"Expected 60 bilingual pairs for {model}")
        possible = {(0, 0)}
        complete = 0
        for labels in model_pairs:
            en, az = labels["en"], labels["az"]
            complete += en in LABELS and az in LABELS
            increments = set(possible_discordances(en, az))
            possible = {(b + db, c + dc) for b, c in possible for db, dc in increments}
        values = [exact_mcnemar(b, c) for b, c in possible]
        row = {
            "model": model,
            "complete_pairs": complete,
            "uncertain_pairs": 60 - complete,
            "min_p": f"{min(values):.15g}",
            "max_p": f"{max(values):.15g}",
        }
        output.append(row)
        print(f"{model}: complete={complete}, uncertain_pairs={60-complete}, p=[{min(values):.12g}, {max(values):.12g}]")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
        print(f"Wrote aggregate results to {args.out}")


if __name__ == "__main__":
    main()

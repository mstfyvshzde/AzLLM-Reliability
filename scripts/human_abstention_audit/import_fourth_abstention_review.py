#!/usr/bin/env python3
"""Import stage-4 abstention judgments without imputing unresolved labels.

This verifies source text and provenance alignment across all four rounds,
preserves the reviewer files and existing 130 stage-3 final decisions,
records the stage-4 responses, and generates partially identified
human-audited abstention-rate intervals for each model-language stratum.

Reviewer independence, identity, and annotation quality cannot be confirmed
by the script; all statements are descriptive about supplied labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VALID = {'yes', 'no', 'uncertain'}
BLIND = ['audit_id', 'language', 'question', 'model_response', 'final_adjudication', 'reviewer_notes']


def load(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing required file: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f'Missing header in {path}')
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f'Unexpected extra columns in {path}')
    if not rows:
        raise ValueError(f'No rows in {path}')
    return rows, list(reader.fieldnames)


def by_id(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    out = {}
    for row in rows:
        key = (row.get('audit_id') or '').strip()
        if not key:
            raise ValueError(f'Missing audit_id in {name}')
        if key in out:
            raise ValueError(f'Duplicate audit_id in {name}: {key}')
        out[key] = row
    return out


def label(row: dict[str, str], field: str) -> str:
    return (row.get(field) or '').strip().lower()


def save_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix='.write_', suffix='.csv')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='raise')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    except BaseException:
        try: os.unlink(temp)
        except OSError: pass
        raise


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix='.write_', suffix='.json')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.replace(temp, path)
    except BaseException:
        try: os.unlink(temp)
        except OSError: pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--fourth-file', type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_dir.expanduser().resolve()
    fourth_file = args.fourth_file.expanduser().resolve()
    rr1, _ = load(root / 'reviewer1_blind.csv')
    rr2, _ = load(root / 'reviewer2_blind.csv')
    rr3, _ = load(root / 'THIRD_REVIEW_SOURCE_PRESERVED.csv')
    fourth, four_columns = load(fourth_file)
    adj, adj_columns = load(root / 'ADJUDICATION_REQUIRED.csv')
    admin, _ = load(root / 'ADMIN_DO_NOT_SEND_TO_REVIEWERS.csv')
    if list(four_columns) != BLIND:
        raise ValueError(f'Fourth review headers must be {BLIND}, found {four_columns}')
    n = [len(rr1), len(rr2), len(rr3), len(fourth), len(adj), len(admin)]
    if n != [360, 360, 172, 42, 172, 360]:
        raise ValueError(f'Unexpected lengths: {n}; expected [360, 360, 172, 42, 172, 360]')
    r1 = by_id(rr1, 'reviewer1')
    r2 = by_id(rr2, 'reviewer2')
    r3 = by_id(rr3, 'third')
    r4 = by_id(fourth, 'fourth')
    ad = by_id(adj, 'adjudication')
    mapping = by_id(admin, 'ADMIN')
    if not (set(r1) == set(r2) == set(mapping)):
        raise ValueError('Reviewer 1, 2, and ADMIN IDs differ')
    expected_third = {
        key for key in r1
        if label(r1[key], 'correct_abstention') != label(r2[key], 'correct_abstention')
        or label(r1[key], 'correct_abstention') == 'uncertain'
    }
    if not (set(ad) == set(r3) == expected_third):
        raise ValueError('Adjudication, third review, and initial disagree/uncertain IDs do not match')
    expected_fourth = {key for key, value in r3.items() if label(value, 'final_adjudication') == 'uncertain'}
    if set(r4) != expected_fourth:
        raise ValueError('Fourth-review ID set does not match exactly the 42 third-review uncertain records')
    for key in r1:
        for source in (r1[key], r2[key]):
            if label(source, 'correct_abstention') not in VALID:
                raise ValueError(f'Invalid reviewer 1/2 label {key}')
        for col in ('language', 'question', 'model_response'):
            if r1[key][col] != r2[key][col]:
                raise ValueError(f'Original reviewers have different source content at {key} {col}')
    for key, row in r3.items():
        if label(row, 'final_adjudication') not in VALID:
            raise ValueError(f'Invalid third-review decision at {key}')
        for col in ('language', 'question', 'model_response'):
            if row[col] != r1[key][col]:
                raise ValueError(f'Third source content changed at {key} {col}')
        prev = label(ad[key], 'final_adjudication')
        third = label(row, 'final_adjudication')
        if third == 'uncertain':
            if prev not in ('',):
                raise ValueError(f'Adjudication already filled but third uncertain at {key}: {prev}')
        elif prev != third:
            raise ValueError(f'Previously adjudicated third decision changed at {key}: {prev} != {third}')
    count = Counter()
    for key, row in r4.items():
        for col in ('language', 'question', 'model_response'):
            if row[col] != r1[key][col] or row[col] != r3[key][col]:
                raise ValueError(f'Fourth source content mismatch at {key} {col}')
        value = label(row, 'final_adjudication')
        if value not in VALID:
            raise ValueError(f'Invalid fourth decision at {key}: {value!r}')
        if not row.get('reviewer_notes', '').strip():
            raise ValueError(f'Missing fourth reviewer note at {key}')
        count[value] += 1
    if count != Counter({'yes': 15, 'no': 2, 'uncertain': 25}):
        raise ValueError(f'Unexpected fourth-label counts: {dict(count)}')

    # The first two reviewers' unequivocal agreements, stage-3 decisions,
    # and stage-4 decisions are the only allowable bases for final labels.
    outcome = {}
    stage = {}
    for key in r1:
        x, y = label(r1[key], 'correct_abstention'), label(r2[key], 'correct_abstention')
        if x == y and x in ('yes', 'no'):
            outcome[key], stage[key] = x, 'initial_agreement'
        elif key in r4:
            fourth_value = label(r4[key], 'final_adjudication')
            if fourth_value in ('yes', 'no'):
                outcome[key], stage[key] = fourth_value, 'fourth_review'
            else:
                stage[key] = 'unresolved_fourth_uncertain'
        else:
            third_value = label(r3[key], 'final_adjudication')
            if third_value not in ('yes', 'no'):
                raise ValueError(f'Unexpected unresolved third decision at {key}')
            outcome[key], stage[key] = third_value, 'third_review'
    resolved = Counter(stage.values())
    if resolved != Counter({'initial_agreement': 188, 'third_review': 130,
                            'fourth_review': 17, 'unresolved_fourth_uncertain': 25}):
        raise ValueError(f'Unexpected stages {dict(resolved)}')

    changed = []
    for row in adj:
        key = row['audit_id']
        new_row = dict(row)
        if key in r4 and label(r4[key], 'final_adjudication') in ('yes','no'):
            if label(row, 'final_adjudication') != '':
                raise ValueError(f'Refusing to overwrite existing final decision at {key}')
            new_row['final_adjudication'] = label(r4[key], 'final_adjudication')
        changed.append(new_row)
    unresolved = [x for x in fourth if label(x, 'final_adjudication') == 'uncertain']
    if len(unresolved) != 25:
        raise ValueError('Unexpected unresolved count')
    by_group = defaultdict(lambda: Counter())
    for key, entry in mapping.items():
        group = (entry['model'], entry['language'])
        by_group[group]['n'] += 1
        if key in outcome:
            by_group[group][outcome[key]] += 1
        else:
            by_group[group]['uncertain'] += 1
        by_group[group]['rule_yes'] += (entry.get('rule_based_outcome') == 'correct_abstention')
    summary = []
    for (model, lang), cnt in sorted(by_group.items()):
        n = cnt['n']
        if n != 60: raise ValueError(f'Expected n=60 per group, got {n} for {model}/{lang}')
        yes, no, unknown = cnt['yes'], cnt['no'], cnt['uncertain']
        if yes + no + unknown != 60: raise ValueError('Group counts do not sum to 60')
        summary.append({
            'model': model, 'language': lang, 'n': n,
            'resolved_yes': yes, 'resolved_no': no, 'unresolved': unknown,
            'min_human_abstention_rate': f'{yes/n:.8f}',
            'max_human_abstention_rate': f'{(yes+unknown)/n:.8f}',
            'original_rule_abstentions': cnt['rule_yes'],
            'original_rule_abstention_rate': f'{cnt["rule_yes"]/n:.8f}',
        })
    if sum(x['unresolved'] for x in summary) != 25:
        raise ValueError('Unresolved group count does not sum to 25')

    # Write only after all validations have succeeded.
    prior = root / 'ADJUDICATION_REQUIRED_before_fourth_import.csv'
    if prior.exists():
        raise FileExistsError(f'Backup file already exists: {prior}. Review before retrying.')
    shutil.copy2(root/'ADJUDICATION_REQUIRED.csv', prior)
    save_csv(root/'FOURTH_REVIEW_SOURCE_PRESERVED.csv', BLIND, fourth)
    save_csv(root/'ADJUDICATION_REQUIRED.csv', adj_columns, changed)
    save_csv(root/'UNRESOLVED_25_BLIND.csv', BLIND, [{
        **{k: item[k] for k in ('audit_id', 'language', 'question', 'model_response')},
        'final_adjudication': '', 'reviewer_notes': ''
    } for item in unresolved])
    save_csv(root/'HUMAN_ABSTENTION_BOUNDS_25_UNRESOLVED.csv', [
        'model','language','n','resolved_yes','resolved_no','unresolved',
        'min_human_abstention_rate','max_human_abstention_rate',
        'original_rule_abstentions','original_rule_abstention_rate'], summary)
    report = {
        'status':'partially human adjudicated; 25 ambiguous/unresolved labels',
        'reviewer_identity_and_independence_independently_verified':False,
        'fourth_source_file':str(fourth_file.name),
        'fourth_labels':dict(count),
        'provenance_stage_counts':dict(resolved),
        'total_outputs':360,
        'resolved_yes_no':len(outcome),
        'unresolved':len(unresolved),
        'by_model_language_bounds':summary,
        'notes': 'Do not silently impute uncertain human labels. Rates are descriptive identified bounds under the available reviewer labels, not uncertainty CIs or fully adjudicated estimates. Never replace frozen rule-based model outputs.'
    }
    save_json(root/'FOURTH_REVIEW_IMPORT_SUMMARY.json', report)
    print('SUCCESS: 42 fourth-review judgments exactly match the prior uncertain set and original prompts/responses.')
    print('Fourth decisions: 15 yes, 2 no, 25 uncertain (all reviewer notes preserved).')
    print('RESOLVED: 335/360; UNRESOLVED: 25/360 (AZ 20, EN 5).')
    print('Partial-identification bounds by model and language:')
    for row in summary:
        print(f"  {row['model']:18s} {row['language']} n=60; known yes={row['resolved_yes']:2d}, "
              f"no={row['resolved_no']:2d}, uncertain={row['unresolved']:2d}; "
              f"rate=[{float(row['min_human_abstention_rate']):.1%}, "
              f"{float(row['max_human_abstention_rate']):.1%}]")
    print('Files: FOURTH_REVIEW_SOURCE_PRESERVED.csv, ADJUDICATION_REQUIRED.csv,')
    print('       UNRESOLVED_25_BLIND.csv, HUMAN_ABSTENTION_BOUNDS_25_UNRESOLVED.csv,')
    print('       FOURTH_REVIEW_IMPORT_SUMMARY.json')
    print('Backup: ADJUDICATION_REQUIRED_before_fourth_import.csv')
    print('CAUTION: This is not a completed 360/360 definitive human audit. Reviewer independence and judgments are not source-verified.')

if __name__ == '__main__':
    main()

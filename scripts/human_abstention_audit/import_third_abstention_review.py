#!/usr/bin/env python3
"""Safely import third-stage human abstention judgments (no synthetic labels).

This script does NOT claim reviewer identity/independence is independently verified.
It never modifies the original reviewer files or the original model predictions.
Only unambiguous yes/no third judgments are imported into ADJUDICATION_REQUIRED.csv.
Uncertain judgments remain unresolved. Stage-3 provenance and blind packet are saved.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

YESNO={"yes","no"}
VALID=YESNO|{"uncertain"}
BLIND_COLS=["audit_id","language","question","model_response","final_adjudication","reviewer_notes"]

def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None: raise ValueError(f"Missing header: {path}")
        result=list(reader)
    if not result: raise ValueError(f"Empty file: {path}")
    return result

def index(rows,name):
    d={}
    for r in rows:
        i=(r.get("audit_id") or "").strip()
        if not i: raise ValueError(f"Empty audit_id in {name}")
        if i in d: raise ValueError(f"Duplicate audit_id {i} in {name}")
        d[i]=r
    return d

def save_atomic(path,fieldnames,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=".tmp_",suffix=".csv",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(temp,path)
    except Exception:
        try: os.unlink(temp)
        except OSError: pass
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-dir",type=Path,required=True)
    p.add_argument("--third-file",type=Path,required=True)
    a=p.parse_args()
    root=a.audit_dir.expanduser().resolve()
    src=a.third_file.expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    if not src.is_file():raise FileNotFoundError(src)
    rr1=read(root/"reviewer1_blind.csv")
    rr2=read(root/"reviewer2_blind.csv")
    old=read(root/"ADJUDICATION_REQUIRED.csv")
    third=read(src)
    if len(rr1)!=360 or len(rr2)!=360 or len(old)!=172 or len(third)!=172:
        raise ValueError(f"Unexpected lengths: r1={len(rr1)} r2={len(rr2)} pending={len(old)} third={len(third)}; expected 360/360/172/172")
    r1=index(rr1,"reviewer1")
    r2=index(rr2,"reviewer2")
    prev=index(old,"ADJUDICATION_REQUIRED")
    td=index(third,"third")
    if set(r1)!=set(r2):raise ValueError("reviewer1 and reviewer2 IDs do not match")
    if set(prev)!=set(td):
        raise ValueError(f"Third IDs do not match adjudication packet: missing={sorted(set(prev)-set(td))[:8]}, unexpected={sorted(set(td)-set(prev))[:8]}")
    expected={}
    for i in r1:
        x=(r1[i].get("correct_abstention") or "").strip().lower()
        y=(r2[i].get("correct_abstention") or "").strip().lower()
        if x not in VALID or y not in VALID: raise ValueError(f"Invalid reviewer label {i}: {x!r}, {y!r}")
        for key in ("language","question","model_response"):
            if r1[i].get(key)!=r2[i].get(key):raise ValueError(f"Reviewer source content mismatch: {i} {key}")
        if x!=y or x=="uncertain":expected[i]=(x,y)
    if set(expected)!=set(prev):raise ValueError("Existing adjudication packet differs from reviewer labels")
    for i,row in prev.items():
        x,y=expected[i]
        if row.get("reviewer1","").strip().lower()!=x or row.get("reviewer2","").strip().lower()!=y:
            raise ValueError(f"ADJUDICATION_REQUIRED reviewer labels differ at {i}")
    count=Counter()
    updated=[]
    unresolved=[]
    for original in old:
        i=original["audit_id"].strip()
        t=td[i]
        for key in ("language","question","model_response"):
            if t.get(key)!=r1[i].get(key): raise ValueError(f"Third source content mismatch: {i} {key}")
        label=(t.get("final_adjudication") or "").strip().lower()
        if label not in VALID:raise ValueError(f"Invalid third label: {i} {label!r}")
        notes=(t.get("reviewer_notes") or "").strip()
        if label=="uncertain" and not notes:raise ValueError(f"Uncertain third label requires notes: {i}")
        before=(original.get("final_adjudication") or "").strip().lower()
        if before not in ("",label):
            raise ValueError(f"Refusing to overwrite existing adjudication: {i} previously {before!r}, third {label!r}")
        o=dict(original)
        if label in YESNO: o["final_adjudication"]=label
        else:
            o["final_adjudication"]=""
            unresolved.append({"audit_id":i,"language":t["language"],"question":t["question"],"model_response":t["model_response"],"final_adjudication":"","reviewer_notes":""})
        updated.append(o)
        count[label]+=1
    if dict(count)!={"yes":109,"no":21,"uncertain":42}:
        # The script is specific to this inspected source; stop for unexplained changes.
        raise ValueError(f"Unexpected third label totals: {dict(count)}")
    backup=root/"ADJUDICATION_REQUIRED_before_third_import.csv"
    if backup.exists():
        # Preserve the original initial backup across repeat runs.
        if read(backup)!=old and any((r.get('final_adjudication') or '').strip() for r in old):
            pass
    else:shutil.copy2(root/"ADJUDICATION_REQUIRED.csv",backup)
    save_atomic(root/"THIRD_REVIEW_SOURCE_PRESERVED.csv",BLIND_COLS,third)
    save_atomic(root/"FOURTH_REVIEW_BLIND_42.csv",BLIND_COLS,unresolved)
    save_atomic(root/"ADJUDICATION_REQUIRED.csv",list(old[0].keys()),updated)
    report={
        "status":"partially adjudicated; unresolved cases remain",
        "source_file_name":src.name,
        "reviewer_identity_and_independence_independently_verified":False,
        "third_review_total":172,
        "third_review_labels":dict(count),
        "preexisting_agreed_yes_no":188,
        "adjudicated_yes_no_by_third":count["yes"]+count["no"],
        "resolved_total":188+count["yes"]+count["no"],
        "unresolved_total":count["uncertain"],
        "details":"No automatic replacement of uncertain labels. No final model scores should be reported as fully human-adjudicated until remaining cases are handled transparently."
    }
    (root/"THIRD_REVIEW_IMPORT_SUMMARY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUCCESS: 172 third-review judgments matched the original review packet and reviewer source texts.")
    print("Imported: 109 yes + 21 no = 130 decisions; retained 42 uncertain as unresolved.")
    print("Resolved: 318/360; unresolved: 42/360.")
    print("Files: ADJUDICATION_REQUIRED.csv, THIRD_REVIEW_SOURCE_PRESERVED.csv, FOURTH_REVIEW_BLIND_42.csv, THIRD_REVIEW_IMPORT_SUMMARY.json")
    print("Backup: ADJUDICATION_REQUIRED_before_third_import.csv")
    print("CAUTION: Third-review independence and label quality are not independently verified.")

if __name__=="__main__":main()

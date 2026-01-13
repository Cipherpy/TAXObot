#!/usr/bin/env python3
"""
Merge Excel ground-truth with pipeline JSONL outputs.

Requested behavior:
- From Excel take ONLY: Question, Expected_answer
- From JSONL take EVERYTHING ELSE
- Match ONLY on normalized Question text

ADDED:
- Detect and REPORT duplicate questions in JSONL
- DO NOT remove or modify duplicates
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# -----------------------------
# Excel column names
# -----------------------------
EXCEL_Q_COL = "Question"
EXCEL_EXP_COL = "Expected_answer"


# -----------------------------
# Helpers
# -----------------------------
def norm_q(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_get(d: Dict[str, Any], keys: List[str], default=""):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def get_json_question(rec: Dict[str, Any]) -> str:
    return str(safe_get(rec, ["Question", "question", "query", "prompt"], "")).strip()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--print_dups", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load Excel
    df = pd.read_excel(args.excel)
    for c in [EXCEL_Q_COL, EXCEL_EXP_COL]:
        if c not in df.columns:
            raise ValueError(f"Excel missing column: {c}")

    df["_q_norm"] = df[EXCEL_Q_COL].apply(norm_q)

    # ---- Load JSONL
    recs = read_jsonl(Path(args.jsonl))

    # ---- Detect duplicates (NO REMOVAL)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in recs:
        nq = norm_q(get_json_question(r))
        if nq:
            groups.setdefault(nq, []).append(r)

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}

    print(f"JSONL records           : {len(recs)}")
    print(f"Duplicate question keys : {len(dup_groups)}")
    print(f"Extra duplicate rows    : {sum(len(v)-1 for v in dup_groups.values())}")

    if args.print_dups:
        print("\n--- DUPLICATES ---")
        for i, (nq, items) in enumerate(sorted(dup_groups.items(),
                                               key=lambda x: len(x[1]),
                                               reverse=True), 1):
            q = get_json_question(items[0])
            print(f"\n[{i}] count={len(items)}")
            print(f"Question: {q[:200]}")

    # ---- Save duplicates report
    dup_jsonl = []
    dup_csv = []

    for nq, items in dup_groups.items():
        q = get_json_question(items[0])
        ids = [safe_get(r, ["id", "row_id", "qid"], "") for r in items]

        dup_jsonl.append({
            "_q_norm": nq,
            "Question_example": q,
            "count": len(items),
            "ids": ids,
            "records": items
        })

        dup_csv.append({
            "_q_norm": nq,
            "Question_example": q,
            "count": len(items),
            "ids_joined": " | ".join(map(str, ids))
        })

    write_jsonl(outdir / "DUPLICATES.jsonl", dup_jsonl)
    pd.DataFrame(dup_csv).to_csv(outdir / "DUPLICATES.csv", index=False)

    # ---- Index JSONL (unchanged behavior: first seen wins)
    json_by_q = {k: v[0] for k, v in groups.items()}

    # ---- Merge
    merged = []
    rows = []
    missed = 0

    for i, r in df.iterrows():
        q = str(r[EXCEL_Q_COL]).strip()
        exp = str(r[EXCEL_EXP_COL]).strip()
        nq = norm_q(q)

        jr = json_by_q.get(nq)
        if jr is None:
            missed += 1
            rec = {"Question": q, "Expected_answer": exp, "merge_status": "NOT_FOUND"}
        else:
            rec = dict(jr)
            rec["Question"] = q
            rec["Expected_answer"] = exp
            rec["merge_status"] = "OK"
            if nq in dup_groups:
                rec["merge_note"] = f"DUPLICATE_JSONL(count={len(dup_groups[nq])})"

        merged.append(rec)

        rows.append({
            "row": i + 1,
            "Question": q,
            "Expected_answer": exp,
            "Generated_answer": safe_get(rec, ["generated_answer", "final_answer"]),
            "merge_status": rec["merge_status"],
            "merge_note": rec.get("merge_note", "")
        })

    # ---- Write outputs
    write_jsonl(outdir / "MERGED.jsonl", merged)
    pd.DataFrame(rows).to_csv(outdir / "MERGED.csv", index=False)
    pd.DataFrame(rows).to_excel(outdir / "MERGED.xlsx", index=False)

    print(f"\nMissed merges: {missed}/{len(df)}")
    print("Done.")


if __name__ == "__main__":
    main()

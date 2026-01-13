#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import List, Dict, Any, Optional

import pandas as pd


# -------------------------
# Metrics (NO preprocessing)
# -------------------------
def compute_exact_match_raw(pred: str, ref: str) -> float:
    """Strict raw string equality (case/space/newline sensitive)."""
    return 1.0 if pred == ref else 0.0


def simple_tokenize_lower_alnum(s: str) -> List[str]:
    """
    Tokenization used ONLY for token-level metrics (not for cleaning the columns).
    """
    import re
    s = "" if s is None else str(s)
    return re.findall(r"[a-z0-9]+", s.lower())


def compute_token_f1(pred: str, ref: str) -> float:
    pt = simple_tokenize_lower_alnum(pred)
    rt = simple_tokenize_lower_alnum(ref)
    if not pt and not rt:
        return 1.0
    if not pt or not rt:
        return 0.0

    from collections import Counter
    pc = Counter(pt)
    rc = Counter(rt)
    common = sum((pc & rc).values())
    if common == 0:
        return 0.0
    precision = common / len(pt)
    recall = common / len(rt)
    return 2 * precision * recall / (precision + recall)


# -------------------------
# Optional scorers
# -------------------------
def try_compute_rouge(preds: List[str], refs: List[str]) -> Optional[List[Dict[str, float]]]:
    try:
        from rouge_score import rouge_scorer
    except Exception as e:
        print(f"[WARN] ROUGE skipped (install 'rouge-score'). Reason: {e}")
        return None

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)  # (target, prediction)
        rows.append({
            "rouge1_f": s["rouge1"].fmeasure,
            "rouge2_f": s["rouge2"].fmeasure,
            "rougeL_f": s["rougeL"].fmeasure,
        })
    return rows


def ensure_nltk():
    try:
        import nltk
        from nltk.data import find

        needed = [
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
            ("tokenizers/punkt", "punkt"),
        ]
        for path, pkg in needed:
            try:
                find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception:
        pass


def compute_bleu_sentence(preds: List[str], refs: List[str]) -> List[float]:
    try:
        ensure_nltk()
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
    except Exception as e:
        print(f"[WARN] BLEU skipped (install 'nltk'). Reason: {e}")
        return [0.0] * len(preds)

    scores = []
    for p, r in zip(preds, refs):
        pt = simple_tokenize_lower_alnum(p)
        rt = simple_tokenize_lower_alnum(r)
        if not pt or not rt:
            scores.append(0.0)
        else:
            scores.append(float(sentence_bleu([rt], pt, smoothing_function=smooth)))
    return scores


def compute_meteor(preds: List[str], refs: List[str]) -> List[float]:
    try:
        ensure_nltk()
        from nltk.translate.meteor_score import meteor_score
    except Exception as e:
        print(f"[WARN] METEOR skipped (install 'nltk'). Reason: {e}")
        return [0.0] * len(preds)

    scores = []
    for p, r in zip(preds, refs):
        pt = simple_tokenize_lower_alnum(p)
        rt = simple_tokenize_lower_alnum(r)
        if not pt or not rt:
            scores.append(0.0)
        else:
            scores.append(float(meteor_score([rt], pt)))
    return scores


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else 0.0


def add_metrics_block(df: pd.DataFrame, refs: List[str], preds: List[str], prefix: str) -> Dict[str, float]:
    df[f"{prefix}exact_match"] = [compute_exact_match_raw(p, r) for p, r in zip(preds, refs)]
    df[f"{prefix}token_f1"] = [compute_token_f1(p, r) for p, r in zip(preds, refs)]

    rouge_rows = try_compute_rouge(preds, refs)
    if rouge_rows is not None and len(rouge_rows) > 0:
        df[f"{prefix}rouge1_f"] = [row["rouge1_f"] for row in rouge_rows]
        df[f"{prefix}rouge2_f"] = [row["rouge2_f"] for row in rouge_rows]
        df[f"{prefix}rougeL_f"] = [row["rougeL_f"] for row in rouge_rows]
    else:
        df[f"{prefix}rouge1_f"] = None
        df[f"{prefix}rouge2_f"] = None
        df[f"{prefix}rougeL_f"] = None

    df[f"{prefix}bleu"] = compute_bleu_sentence(preds, refs)
    df[f"{prefix}meteor"] = compute_meteor(preds, refs)

    summary = {
        f"{prefix}exact_match": mean(df[f"{prefix}exact_match"].tolist()),
        f"{prefix}token_f1": mean(df[f"{prefix}token_f1"].tolist()),
        f"{prefix}bleu": mean(df[f"{prefix}bleu"].tolist()),
        f"{prefix}meteor": mean(df[f"{prefix}meteor"].tolist()),
        f"{prefix}rouge1_f": mean([x for x in df[f"{prefix}rouge1_f"].tolist() if x is not None])
        if df[f"{prefix}rouge1_f"].notna().any()
        else None,
        f"{prefix}rouge2_f": mean([x for x in df[f"{prefix}rouge2_f"].tolist() if x is not None])
        if df[f"{prefix}rouge2_f"].notna().any()
        else None,
        f"{prefix}rougeL_f": mean([x for x in df[f"{prefix}rougeL_f"].tolist() if x is not None])
        if df[f"{prefix}rougeL_f"].notna().any()
        else None,
    }
    return summary


# -------------------------
# Robust CSV read (fixes UnicodeDecodeError 0x96)
# -------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Compare two columns directly (NO preprocessing, NO species extraction).")
    ap.add_argument("--input_csv", required=True)

    ap.add_argument("--output_csv", required=True, help="Scored CSV output path")
    ap.add_argument("--output_xlsx", required=True, help="Excel output path")
    ap.add_argument("--summary_json", default=None, help="Optional summary JSON path")

    ap.add_argument("--expected_col", default="Expected_Answer")
    ap.add_argument("--generated_col", default="GPT5_Answers")

    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(args.input_csv)

    df = safe_read_csv(args.input_csv)

    if args.expected_col not in df.columns:
        raise ValueError(f"Missing expected_col={args.expected_col}. Found: {list(df.columns)}")
    if args.generated_col not in df.columns:
        raise ValueError(f"Missing generated_col={args.generated_col}. Found: {list(df.columns)}")

    if args.limit:
        df = df.head(args.limit).copy()

    # NO preprocessing: keep raw strings, just avoid NaN/None issues
    df[args.expected_col] = df[args.expected_col].fillna("").astype(str)
    df[args.generated_col] = df[args.generated_col].fillna("").astype(str)

    refs = df[args.expected_col].tolist()
    preds = df[args.generated_col].tolist()

    summary = add_metrics_block(df, refs, preds, prefix="direct_")

    # Save
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")

    os.makedirs(os.path.dirname(args.output_xlsx) or ".", exist_ok=True)
    df.to_excel(args.output_xlsx, index=False)

    out_summary = {
        "rows": int(len(df)),
        "columns_used": {"expected_col": args.expected_col, "generated_col": args.generated_col},
        "means": summary,
        "notes": {
            "preprocessing": "None. Exact match uses strict raw equality.",
            "token_metrics": "Token-based metrics (F1/BLEU/METEOR/ROUGE) tokenize internally for scoring only."
        },
    }

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(out_summary, f, indent=2)

    print("✅ Done")
    print("Output CSV:", args.output_csv)
    print("Output XLSX:", args.output_xlsx)
    if args.summary_json:
        print("Summary JSON:", args.summary_json)
    print("Mean metrics:", out_summary["means"])


if __name__ == "__main__":
    main()

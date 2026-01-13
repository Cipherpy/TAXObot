#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import List, Dict, Any, Optional

import pandas as pd


# -------------------------
# Text normalization
# -------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def simple_tokenize(s: str) -> List[str]:
    s = normalize_text(s).lower()
    return re.findall(r"[a-z0-9]+", s)


# -------------------------
# Extract ONLY text BEFORE "Support:" (exclude "Support:" and everything after)
# -------------------------
def extract_before_support(text: str) -> str:
    """
    Returns the substring strictly BEFORE the first occurrence of 'Support:' (case-insensitive).
    Excludes 'Support:' itself and everything after it.

    Example:
      "Answer: X. Support: blah blah" -> "Answer: X."
      "Support: blah blah" -> ""   (nothing before Support)
      No Support: -> full text

    Also removes a leading 'assistant' prefix if present.
    """
    t = "" if text is None else str(text)
    t = t.replace("\u00a0", " ").strip()

    # Remove leading "assistant" label if present
    t = re.sub(r"^\s*assistant\s*[:\-]?\s*", "", t, flags=re.IGNORECASE)

    m = re.search(r"\bSupport\s*:\s*", t, flags=re.IGNORECASE)
    if m:
        t = t[:m.start()].strip()  # STRICTLY before "Support:"
    return normalize_text(t)


# -------------------------
# NLTK for METEOR (optional downloads)
# -------------------------
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


# -------------------------
# Metrics (robust + lightweight)
# -------------------------
def compute_exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0


def compute_token_f1(pred: str, ref: str) -> float:
    pt = simple_tokenize(pred)
    rt = simple_tokenize(ref)
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


def try_compute_rouge(preds: List[str], refs: List[str]) -> Optional[List[Dict[str, float]]]:
    """
    Optional: requires rouge-score (module rouge_score).
    """
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


def compute_bleu_sentence(preds: List[str], refs: List[str]) -> List[float]:
    """
    Optional: needs nltk. If not available, returns 0.0s.
    """
    try:
        ensure_nltk()
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
    except Exception as e:
        print(f"[WARN] BLEU skipped (install 'nltk'). Reason: {e}")
        return [0.0] * len(preds)

    scores = []
    for p, r in zip(preds, refs):
        pt = simple_tokenize(p)
        rt = simple_tokenize(r)
        if not pt or not rt:
            scores.append(0.0)
        else:
            scores.append(float(sentence_bleu([rt], pt, smoothing_function=smooth)))
    return scores


def compute_meteor(preds: List[str], refs: List[str]) -> List[float]:
    """
    Optional: needs nltk + wordnet resources. If not available, returns 0.0s.
    """
    try:
        ensure_nltk()
        from nltk.translate.meteor_score import meteor_score
    except Exception as e:
        print(f"[WARN] METEOR skipped (install 'nltk'). Reason: {e}")
        return [0.0] * len(preds)

    scores = []
    for p, r in zip(preds, refs):
        pt = simple_tokenize(p)
        rt = simple_tokenize(r)
        if not pt or not rt:
            scores.append(0.0)
        else:
            scores.append(float(meteor_score([rt], pt)))
    return scores


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else 0.0


def add_metrics_block(
    df: pd.DataFrame,
    refs: List[str],
    preds: List[str],
    prefix: str
) -> Dict[str, float]:
    """
    Adds metric columns to df with a prefix, returns mean summary for that block.
    """
    df[f"{prefix}exact_match"] = [compute_exact_match(p, r) for p, r in zip(preds, refs)]
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
        if df[f"{prefix}rouge1_f"].notna().any() else None,
        f"{prefix}rouge2_f": mean([x for x in df[f"{prefix}rouge2_f"].tolist() if x is not None])
        if df[f"{prefix}rouge2_f"].notna().any() else None,
        f"{prefix}rougeL_f": mean([x for x in df[f"{prefix}rougeL_f"].tolist() if x is not None])
        if df[f"{prefix}rougeL_f"].notna().any() else None,
    }
    return summary


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Evaluate Generated_answer text BEFORE 'Support:' vs Expected_answer.")
    ap.add_argument("--input_csv", required=True)

    # outputs
    ap.add_argument("--output_csv", required=True, help="Scored CSV output path")
    ap.add_argument("--output_xlsx", required=True, help="Excel output path")
    ap.add_argument("--summary_json", default=None, help="Optional summary JSON path")

    # columns
    ap.add_argument("--expected_col", default="Expected_answer")
    ap.add_argument("--generated_col", default="Generated_answer")

    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(args.input_csv)

    df = pd.read_csv(args.input_csv)

    if args.expected_col not in df.columns:
        raise ValueError(f"Missing expected_col={args.expected_col}. Found: {list(df.columns)}")
    if args.generated_col not in df.columns:
        raise ValueError(f"Missing generated_col={args.generated_col}. Found: {list(df.columns)}")

    if args.limit:
        df = df.head(args.limit).copy()

    # Ensure strings
    df[args.expected_col] = df[args.expected_col].fillna("").astype(str)
    df[args.generated_col] = df[args.generated_col].fillna("").astype(str)

    # Extract generated content strictly before Support:
    df["Generated_before_support"] = df[args.generated_col].apply(extract_before_support)

    # ---------- Metrics: Generated_before_support vs Expected_answer ----------
    refs = [normalize_text(x) for x in df[args.expected_col].tolist()]
    preds = [normalize_text(x) for x in df["Generated_before_support"].tolist()]
    summary_means = add_metrics_block(df, refs, preds, prefix="eval_")

    # Save outputs
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    os.makedirs(os.path.dirname(args.output_xlsx) or ".", exist_ok=True)
    df.to_excel(args.output_xlsx, index=False)

    summary = {
        "rows": int(len(df)),
        "columns_used": {"expected_col": args.expected_col, "generated_col": args.generated_col},
        "means": summary_means,
        "generated_before_support_empty_count": int(
            (df["Generated_before_support"].astype(str).str.strip() == "").sum()
        ),
    }

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("✅ Done")
    print("Output CSV:", args.output_csv)
    print("Output XLSX:", args.output_xlsx)
    if args.summary_json:
        print("Summary JSON:", args.summary_json)
    print("Mean metrics:", summary["means"])


if __name__ == "__main__":
    main()

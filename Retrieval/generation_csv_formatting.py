#!/usr/bin/env python3
"""
Convert JSONL to CSV with columns:
  - Question
  - Expected_answer
  - Generated_answer  (extracted from final_answer)

Usage:
  python jsonl_to_csv_extract_answer.py \
      --input merged.jsonl \
      --output merged_extracted.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


# -----------------------------
# Answer extraction
# -----------------------------
ANSWER_RE = re.compile(
    r"(?is)"                      # ignore case + dot matches newline
    r"(?:^|\n)\s*(?:assistant\s*)?" # optional "assistant"
    r"(?:Answer\s*:\s*)"           # "Answer:"
    r"(.*?)"                       # capture answer body (lazy)
    r"(?=\n\s*(?:Explanation|Citations)\s*:|\Z)"  # stop before next section or end
)

def extract_generated_answer(final_answer: Any) -> str:
    """
    Extracts the text after 'Answer:' from final_answer.
    Stops at 'Explanation:' or 'Citations:' if present.
    Falls back to cleaned full text if pattern not found.
    """
    if final_answer is None:
        return ""
    text = str(final_answer).strip()
    if not text:
        return ""

    m = ANSWER_RE.search(text)
    if m:
        ans = m.group(1).strip()
        # Collapse whitespace but keep readable spacing
        ans = re.sub(r"\s+", " ", ans).strip()
        return ans

    # Fallback: sometimes final_answer is just "Answer: X" without sections, or plain text
    # Try a simpler one-liner
    m2 = re.search(r"(?is)\bAnswer\s*:\s*(.+)$", text)
    if m2:
        ans = m2.group(1).strip()
        ans = re.sub(r"\s+", " ", ans).strip()
        return ans

    # Last fallback: return the whole thing (trimmed), but still whitespace-normalized
    return re.sub(r"\s+", " ", text).strip()


# -----------------------------
# IO
# -----------------------------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .jsonl file")
    ap.add_argument("--output", required=True, help="Output .csv file")
    ap.add_argument("--q_col", default="Question", help="Question field name in JSONL")
    ap.add_argument("--exp_col", default="Expected_answer", help="Expected answer field name in JSONL")
    ap.add_argument("--final_col", default="final_answer", help="Final answer field name in JSONL")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["Question", "Expected_answer", "Generated_answer"])
        writer.writeheader()

        n = 0
        for line_no, row in read_jsonl(in_path):
            q = row.get(args.q_col, "")
            exp = row.get(args.exp_col, "")
            final = row.get(args.final_col, "")

            gen = extract_generated_answer(final)

            writer.writerow({
                "Question": q,
                "Expected_answer": exp,
                "Generated_answer": gen
            })
            n += 1

    print(f"✅ Wrote {n} rows to: {out_path}")


if __name__ == "__main__":
    main()

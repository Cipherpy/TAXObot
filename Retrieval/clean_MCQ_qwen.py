#!/usr/bin/env python3
"""
Extract ONLY the Answer/Explanation/Citations block
that appears AFTER '\\n\\nassistant\\n'
and replace final_answer with it.
"""

import json
import re
from pathlib import Path

INPUT_JSONL = "/home/reshma/TAXObot/Retrieval_Ablation/results_ab3/MCQ_merged_ab3_qwen.jsonl"
OUTPUT_JSONL = "/home/reshma/TAXObot/Retrieval_Ablation/ablation1/cleaned_MCQ_qwen.jsonl"

PATTERN = re.compile(
    r"\n\nassistant\n"
    r"(Answer:\s*.*?\nExplanation:\s*.*?\nCitations:\s*\[.*?\])",
    re.DOTALL
)

def extract_after_assistant(text: str) -> str:
    if not isinstance(text, str):
        return ""

    match = PATTERN.search(text)
    return match.group(1).strip() if match else ""

def main():
    in_path = Path(INPUT_JSONL)
    out_path = Path(OUTPUT_JSONL)

    missing = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            row = json.loads(line)
            extracted = extract_after_assistant(row.get("final_answer", ""))

            row["final_answer"] = extracted

            if not extracted:
                missing += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Cleaned JSONL written to: {out_path}")
    if missing:
        print(f"⚠️ Rows with no assistant-answer block found: {missing}")

if __name__ == "__main__":
    main()

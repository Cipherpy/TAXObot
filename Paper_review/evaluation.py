#!/usr/bin/env python3
"""
Score RAG outputs using 4 metrics (0–5 each), total 0–20.

Fix included:
- Escape curly braces in JUDGE_PROMPT so .format() doesn't crash (KeyError).

Also:
- Extract ONLY "Answer:" section from final_answer for evaluation.
- Incremental CSV + JSONL writing.
- Use Qwen2.5-32B-Instruct as judge.
- Use chunking data from each row (rank + text) as Retrieved Context.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# CONFIG
# -----------------------------
JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 260

MAX_CHUNK_CHARS_EACH = 1200
MAX_TOTAL_CONTEXT_CHARS = 9000


# -----------------------------
# IO helpers
# -----------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


# -----------------------------
# Answer extraction (Answer: ... only)
# -----------------------------
ANSWER_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*Answer\s*:\s*(.*?)(?=\n\s*(Support|Citation|Citations|Evidence|Context)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)

def extract_answer_only(text: str) -> str:
    """
    Your final_answer contains:
      ---
      Answer: ...
      Support: ...
      Citations: ...
    We only want the Answer content.
    """
    if not text:
        return ""
    t = str(text).strip()

    m = ANSWER_BLOCK_RE.search(t)
    if m:
        ans = m.group(1).strip()
        ans = re.sub(r"\s+", " ", ans).strip()
        return ans

    # fallback: if starts with "Answer ..."
    if t.lower().startswith("answer"):
        t2 = re.sub(r"^answer\s*[-–:]\s*", "", t, flags=re.IGNORECASE).strip()
        return re.sub(r"\s+", " ", t2).strip()

    return re.sub(r"\s+", " ", t).strip()


# -----------------------------
# Retrieval context extraction (prefer row chunk text)
# -----------------------------
def extract_retrieval_chunks(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Preferred: use chunk text present in the same row.
    Tries common layouts:
    A) rec["retrieval"]["chunks"] = [{"rank","text","metadata":...}, ...]
    B) rec["chunks"] = [{"rank","text","metadata":...}, ...]
    C) rec["used_sources_json"] or "sources" or "used_sources" (stringified JSON list)
       e.g. [{"rank":..,"preview":..,"metadata":..}, ...]
    D) Only chunk_ids exist -> no grounding evidence text.
    """
    chunks: List[Dict[str, Any]] = []

    # A
    r_chunks = safe_get(rec, ["retrieval", "chunks"], None)
    if isinstance(r_chunks, list) and r_chunks:
        for c in r_chunks:
            if not isinstance(c, dict):
                continue
            text = str(c.get("text", "")).strip()
            md = c.get("metadata", {}) if isinstance(c.get("metadata", {}), dict) else {}
            rank = c.get("rank", None)
            chunks.append({"rank": rank, "text": text, "metadata": md})
        return chunks

    # B
    if isinstance(rec.get("chunks"), list) and rec["chunks"]:
        for i, c in enumerate(rec["chunks"], start=1):
            if isinstance(c, dict):
                md = c.get("metadata", {}) if isinstance(c.get("metadata", {}), dict) else {}
                chunks.append({
                    "rank": c.get("rank", i),
                    "text": str(c.get("text", "")).strip(),
                    "metadata": md,
                })
            else:
                chunks.append({"rank": i, "text": str(c).strip(), "metadata": {}})
        return chunks

    # C (stringified sources)
    for key in ["used_sources_json", "used_sources", "sources"]:
        if key in rec and rec[key]:
            try:
                obj = rec[key]
                if isinstance(obj, str):
                    obj = json.loads(obj)
                if isinstance(obj, list):
                    for i, s in enumerate(obj, start=1):
                        if not isinstance(s, dict):
                            continue
                        txt = s.get("text") or s.get("preview") or ""
                        md = s.get("metadata", {}) if isinstance(s.get("metadata", {}), dict) else {}
                        chunks.append({
                            "rank": s.get("rank", i),
                            "text": str(txt).strip(),
                            "metadata": md,
                        })
                    if chunks:
                        return chunks
            except Exception:
                pass

    # D (no text available)
    ids = rec.get("retrieved_chunk_ids")
    if isinstance(ids, list) and ids:
        for i, cid in enumerate(ids[:25], start=1):
            chunks.append({"rank": i, "text": "", "metadata": {"chunk_id": cid}})
    return chunks


def build_context_text(chunks: List[Dict[str, Any]]) -> str:
    """
    Build compact retrieved context for the judge from the *row's chunk text*.
    """
    blocks: List[str] = []
    total = 0
    src_no = 0

    for c in chunks:
        src_no += 1
        rank = c.get("rank", src_no)
        md = c.get("metadata") or {}
        chunk_id = md.get("chunk_id", "") or md.get("id", "") or ""
        source_file = md.get("source_file") or md.get("doc_name") or md.get("source") or md.get("file_name") or ""
        title = md.get("title", "") or md.get("document_title", "") or ""

        text = (c.get("text") or "").strip()
        if text:
            text = text[:MAX_CHUNK_CHARS_EACH]
        else:
            text = "(no chunk text available in JSONL; only chunk_id present)"

        block = f"[Source {src_no}] rank={rank} chunk_id={chunk_id} file={source_file} title={title}\n{text}\n"
        if total + len(block) > MAX_TOTAL_CONTEXT_CHARS:
            break
        blocks.append(block)
        total += len(block)

    return "\n".join(blocks).strip() if blocks else "NO_RETRIEVED_CONTEXT_AVAILABLE"


# -----------------------------
# Judge prompt (ESCAPED braces!)
# -----------------------------
JUDGE_PROMPT = """You are an impartial evaluator for a Retrieval-Augmented Generation (RAG) system.

Score the GENERATED_ANSWER using ONLY these four metrics.
Each metric MUST be an integer from 0 to 5.

1) answer_correctness_0_5
5 = fully correct and matches Expected Answer (allow paraphrase; don't require exact wording)
3–4 = mostly correct with minor issues or small missing detail
1–2 = partially correct / some correct facts but incomplete
0 = wrong or contradicts Expected Answer

2) completeness_0_5
5 = covers all key parts from Expected Answer
3–4 = covers most key parts
1–2 = covers few parts
0 = misses most required parts

3) grounding_faithfulness_0_5
5 = fully supported by Retrieved Context; no hallucinations
3–4 = mostly supported with minor unsupported detail
1–2 = weak support
0 = unsupported / hallucinated relative to Retrieved Context

4) relevance_clarity_0_5
5 = focused, minimal fluff, easy to understand
3–4 = mostly clear and relevant
1–2 = verbose or slightly off-target
0 = confusing or off-topic

Rules:
- Use Expected Answer for correctness + completeness.
- Use Retrieved Context for grounding/faithfulness.
- If a claim is not supported by the context, penalize grounding.
- Output VALID JSON ONLY (no markdown, no extra text).

Return JSON:
{{
  "answer_correctness_0_5": 0,
  "completeness_0_5": 0,
  "grounding_faithfulness_0_5": 0,
  "relevance_clarity_0_5": 0,
  "brief_reason": "1-3 short sentences"
}}

QUESTION:
{question}

EXPECTED_ANSWER:
{expected}

GENERATED_ANSWER:
{generated}

RETRIEVED_CONTEXT:
{context}
"""


def extract_first_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return {}


def clamp_int(x: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(x)
    except Exception:
        v = default
    return max(lo, min(hi, v))


# -----------------------------
# Model loading (Qwen needs pad token handling sometimes)
# -----------------------------
def load_judge_model():
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL, use_fast=True, trust_remote_code=True)

    # ensure pad token exists (important for some Qwen tokenizers)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    if DEVICE != "cuda":
        llm.to(DEVICE)
    llm.eval()
    return tok, llm


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Merged JSONL (must include Question + Expected_answer + generated answer + retrieval)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--limit", type=int, default=0, help="Optional: score only first N records (0 = all)")
    args = ap.parse_args()

    in_path = Path(args.jsonl).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(in_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_jsonl = outdir / f"{in_path.stem}__SCORED_0_20.jsonl"
    out_csv = outdir / f"{in_path.stem}__SCORED_0_20.csv"
    out_xlsx = outdir / f"{in_path.stem}__SCORED_0_20.xlsx"

    tok, llm = load_judge_model()

    csv_fields = [
        "id",
        "Question_type",
        "Question",
        "Expected_answer",
        "Generated_answer_raw",
        "Generated_answer_extracted",
        "answer_correctness_0_5",
        "completeness_0_5",
        "grounding_faithfulness_0_5",
        "relevance_clarity_0_5",
        "total_0_20",
        "normalized_0_1",
        "brief_reason",
    ]

    table_rows_for_xlsx: List[Dict[str, Any]] = []

    with out_jsonl.open("w", encoding="utf-8") as f_jsonl, out_csv.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
        writer.writeheader()

        for idx, rec in enumerate(rows, start=1):
            q = str(rec.get("Question") or rec.get("question") or "").strip()
            expected = str(rec.get("Expected_answer") or rec.get("expected_answer") or "").strip()

            raw_generated = (
                rec.get("generated_answer")
                or rec.get("Generated_answer")
                or rec.get("final_answer")
                or rec.get("answer")
                or ""
            )
            raw_generated = str(raw_generated).strip()

            # IMPORTANT: final_answer contains "Answer: ... Support: ...", so evaluate only Answer:
            generated = extract_answer_only(raw_generated)

            chunks = extract_retrieval_chunks(rec)
            context = build_context_text(chunks)

            prompt = JUDGE_PROMPT.format(
                question=q,
                expected=expected,
                generated=generated,
                context=context,
            )

            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS, padding=True)
            inputs = {k: v.to(llm.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = llm.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            decoded = tok.decode(out[0], skip_special_tokens=True)
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):].strip()

            judge = extract_first_json(decoded)

            c = clamp_int(judge.get("answer_correctness_0_5", 0), 0, 5)
            k = clamp_int(judge.get("completeness_0_5", 0), 0, 5)
            f = clamp_int(judge.get("grounding_faithfulness_0_5", 0), 0, 5)
            r = clamp_int(judge.get("relevance_clarity_0_5", 0), 0, 5)

            total_0_20 = c + k + f + r
            normalized_0_1 = total_0_20 / 20.0

            scored = dict(rec)
            scored["scores"] = {
                "answer_correctness_0_5": c,
                "completeness_0_5": k,
                "grounding_faithfulness_0_5": f,
                "relevance_clarity_0_5": r,
                "total_0_20": total_0_20,
                "normalized_0_1": normalized_0_1,
                "brief_reason": str(judge.get("brief_reason", "")).strip()[:600],
                "judge_model": JUDGE_MODEL,
                "generated_answer_extracted": generated,
            }

            # incremental JSONL
            f_jsonl.write(json.dumps(scored, ensure_ascii=False) + "\n")
            f_jsonl.flush()

            # incremental CSV
            row_csv = {
                "id": rec.get("id", idx),
                "Question_type": rec.get("Question_type", rec.get("question_type", "")),
                "Question": q,
                "Expected_answer": expected,
                "Generated_answer_raw": raw_generated[:2000],
                "Generated_answer_extracted": generated[:2000],
                "answer_correctness_0_5": c,
                "completeness_0_5": k,
                "grounding_faithfulness_0_5": f,
                "relevance_clarity_0_5": r,
                "total_0_20": total_0_20,
                "normalized_0_1": normalized_0_1,
                "brief_reason": scored["scores"]["brief_reason"],
            }
            writer.writerow(row_csv)
            f_csv.flush()

            table_rows_for_xlsx.append(row_csv)

            print(f"[{idx}/{len(rows)}] total_0_20={total_0_20} | C={c} K={k} F={f} R={r}")

    # XLSX at end (since incremental XLSX is painful + slower)
    df = pd.DataFrame(table_rows_for_xlsx)
    df.to_excel(out_xlsx, index=False)

    mean_total = float(df["total_0_20"].mean()) if len(df) else 0.0
    mean_norm = float(df["normalized_0_1"].mean()) if len(df) else 0.0

    print("\n✅ Saved:")
    print(f" - {out_jsonl}")
    print(f" - {out_csv}")
    print(f" - {out_xlsx}")
    print(f"\n📌 Overall mean total: {mean_total:.2f}/20 | mean normalized: {mean_norm:.3f}")


if __name__ == "__main__":
    main()

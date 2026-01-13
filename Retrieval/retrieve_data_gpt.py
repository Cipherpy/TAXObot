#!/usr/bin/env python3
"""
FINAL (OpenAI GPT-4.1-mini): E5 (Chroma) retrieval + OpenAI Responses API generation → incremental JSONL output

✅ Question types ONLY: GQ, MCQ, T1, T2, Yes_No (read from Excel per row OR fixed_type for whole file)
✅ Uses E5 query prefix: "query: "
✅ Strips "passage: " from stored docs (if present)
✅ Retrieves fetch_k then dedupes to EXACT k UNIQUE chunks (by SHIFTED chunk id)
✅ Shifts chunk IDs +1 everywhere (retrieved ids == ids shown to LLM == ids used for citations)
✅ Context shown with explicit chunk ids: [chunk_000123] ...
✅ Uses OpenAI Responses API with system+user messages (no HF chat template needed)
✅ LLM is instructed to cite ONLY valid chunk ids; invalid cites removed
✅ Output is JSONL (incremental append, safe for long runs)

Recommended model:
  --llm_model gpt-4.1-mini

Auth:
  export OPENAI_API_KEY="..."
"""

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from openai import OpenAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()  # requires OPENAI_API_KEY
# -----------------------------
# CONFIG DEFAULTS
# -----------------------------
DEFAULT_E5_MODEL = "intfloat/e5-large-v2"
DEFAULT_OPENAI_MODEL = "gpt-4.1"

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

CHUNK_PAT = re.compile(r"^(chunk_)(\d+)$", re.IGNORECASE)
CITE_RE = re.compile(r"\[(chunk_\d+)\]", re.IGNORECASE)


# -----------------------------
# Utilities
# -----------------------------
def parse_options_cell(x: Any) -> Optional[List[str]]:
    """Accepts Python-list string, JSON list string, or newline-separated text."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None

    # Try list-like formats
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            out = [str(i).strip() for i in v if str(i).strip()]
            return out or None
    except Exception:
        pass

    # Fallback newline split
    if "\n" in s:
        out = [o.strip() for o in s.splitlines() if o.strip()]
        return out or None

    return [s]


def normalize_qtype_to_allowed(label: Any) -> str:
    """
    Excel Question_type mapping to ONLY:
      GQ, MCQ, T1, T2, Yes_No
    """
    s = "" if label is None else str(label).strip()
    if not s:
        return "GQ"

    u = s.upper().replace("-", "_").replace(" ", "_").strip()
    u = re.sub(r"[^\w/]+", "", u)

    if u in {"GQ", "GENERAL", "GEN", "G"}:
        return "GQ"
    if u in {"MCQ", "MCQS", "MULTIPLE_CHOICE", "MULTIPLECHOICE"}:
        return "MCQ"
    if u == "T1":
        return "T1"
    if u == "T2":
        return "T2"
    if u in {"YES_NO", "YESNO", "YES/NO"}:
        return "Yes_No"

    return "GQ"


def normalize_fixed_type(fixed_type: str) -> str:
    """Fixed type must be one of: GQ, MCQ, T1, T2, Yes_No"""
    if fixed_type is None:
        raise ValueError("fixed_type is None")
    return normalize_qtype_to_allowed(fixed_type)


def build_query(question: str, options: Optional[List[str]]) -> str:
    """E5 query prefix required. Options appended for MCQ if provided."""
    q = (question or "").strip()
    if options:
        joined = "\n".join([f"- {o}" for o in options])
        q = f"{q}\nOptions:\n{joined}"
    return QUERY_PREFIX + q


def dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def shift_chunk_id_plus1(cid: Any) -> str:
    """
    Make chunk ids 1-based to match your ground truth.
    Handles:
      - chunk_000000 -> chunk_000001
      - "0" -> "1"
      - 0 -> "1"
    """
    if cid is None:
        return "chunk_UNKNOWN"

    if isinstance(cid, (int, float)) and not pd.isna(cid):
        try:
            return str(int(cid) + 1)
        except Exception:
            return str(cid)

    s = str(cid).strip()

    m = CHUNK_PAT.match(s)
    if m:
        prefix, num = m.group(1), m.group(2)
        width = len(num)
        return f"{prefix}{int(num)+1:0{width}d}"

    if s.isdigit():
        return str(int(s) + 1)

    return s


def choose_base_chunk_id(md: Dict[str, Any], fallback_rank: int) -> str:
    """
    Choose a stable chunk id from metadata.
    IMPORTANT: put your true key FIRST if you know it.
    """
    return str(
        md.get("chunk_number")
        or md.get("chunk_id")
        or md.get("id")
        or md.get("chunk")
        or md.get("source_id")
        or md.get("doc_id")
        or md.get("uuid")
        or f"rank_{fallback_rank}"
    )


def strip_passage_prefix(text: str) -> str:
    t = (text or "").strip()
    if t.startswith(PASSAGE_PREFIX):
        return t[len(PASSAGE_PREFIX):].lstrip()
    return t


def retrieve_exact_k_unique_docs(
    vectordb: Chroma,
    query: str,
    k: int,
    fetch_k: int,
) -> Tuple[List[Any], List[str], int]:
    """
    Fetch fetch_k, then dedupe by SHIFTED chunk id, keep first k unique.
    Returns:
      kept_docs, kept_shifted_ids, raw_count
    """
    docs = vectordb.similarity_search(query, k=fetch_k)
    raw_count = len(docs)

    kept_docs: List[Any] = []
    kept_ids: List[str] = []
    seen = set()

    for rank, d in enumerate(docs, start=1):
        md = d.metadata or {}
        base_id = choose_base_chunk_id(md, fallback_rank=rank)
        shifted_id = shift_chunk_id_plus1(base_id)

        if shifted_id not in seen:
            seen.add(shifted_id)
            kept_docs.append(d)
            kept_ids.append(shifted_id)

        if len(kept_docs) >= k:
            break

    return kept_docs, kept_ids, raw_count


def build_context_with_ids(docs: List[Any], shifted_ids: List[str]) -> str:
    """
    Context with explicit ids: [chunk_000123] <text>
    Ensures the ids shown match retrieved_chunk_ids.
    """
    parts = []
    for d, cid in zip(docs, shifted_ids):
        txt = strip_passage_prefix(d.page_content)
        parts.append(f"[{cid}] {txt}")
    return "\n\n---\n\n".join(parts)


def extract_citations(text: str) -> List[str]:
    return dedup_preserve_order([m.group(1) for m in CITE_RE.finditer(text or "")])


def sanitize_answer_citations(answer: str, valid_set: set) -> Tuple[str, List[str], List[str]]:
    """
    Remove invalid citations to avoid random chunk ids.
    Returns:
      cleaned_answer, valid_citations, invalid_citations
    """
    cites = extract_citations(answer)
    valid = [c for c in cites if c in valid_set]
    invalid = [c for c in cites if c not in valid_set]

    cleaned = answer
    if invalid:
        for c in invalid:
            cleaned = re.sub(rf"\[\s*{re.escape(c)}\s*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[ ]{2,}", " ", cleaned).strip()

    return cleaned, valid, invalid


# -----------------------------
# OpenAI LLM (Responses API)
# -----------------------------
class OpenAIResponsesLLM:
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def generate_chat(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        # Responses API uses max_output_tokens (not max_new_tokens)
        resp = self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_new_tokens,
        )
        # SDK convenience accessor
        return (resp.output_text or "").strip()


# -----------------------------
# Prompt builder (strict citation rules)
# -----------------------------
def build_user_prompt(
    qtype: str,
    question: str,
    options: Optional[List[str]],
    context: str,
    valid_ids: List[str],
) -> str:
    valid_line = ", ".join(valid_ids)
    qt = qtype

    if qt == "MCQ":
        fmt = (
            "Return EXACTLY:\n"
            "Answer: <option letter OR exact option text>\n"
            "Explanation: <1-3 lines>\n"
            "Citations: <[chunk_...]>\n"
        )
    elif qt == "Yes_No":
        fmt = (
            "Return EXACTLY:\n"
            "Answer: Yes/No\n"
            "Justification: <1-2 lines>\n"
            "Citations: <[chunk_...]>\n"
        )
    else:
        fmt = (
            "Return EXACTLY:\n"
            "Answer: <concise>\n"
            "Support: <1-2 lines>\n"
            "Citations: <[chunk_...]>\n"
        )

    opt_block = ""
    if options:
        opt_block = "Options:\n" + "\n".join([f"- {o}" for o in options]) + "\n"

    return (
        "Rules:\n"
        "1) Use ONLY the provided context.\n"
        "2) You may ONLY cite chunk IDs that appear in the context.\n"
        "3) Valid chunk IDs are:\n"
        f"{valid_line}\n"
        "4) Every factual claim must include at least one citation like [chunk_000123].\n"
        "5) If the answer is not in the context, say exactly: Insufficient context.\n"
        "6) Do NOT invent citations.\n\n"
        f"{fmt}\n"
        f"Question type: {qt}\n"
        f"Question: {question.strip()}\n"
        f"{opt_block}\n"
        f"Context:\n{context}\n"
    )


# -----------------------------
# Main (incremental JSONL)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--persist_dir", required=True)
    ap.add_argument("--collection", required=True)

    ap.add_argument("--e5_model", default=DEFAULT_E5_MODEL)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--fetch_k", type=int, default=200)

    # OpenAI model + auth
    ap.add_argument("--llm_model", default=DEFAULT_OPENAI_MODEL)  # e.g., gpt-4.1-mini
    ap.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY", ""))
    ap.add_argument("--openai_base_url", default=os.getenv("OPENAI_BASE_URL", ""))

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--col_qtype", default="Question_type")
    ap.add_argument("--col_question", default="Question")
    ap.add_argument("--col_options", default="Options")

    ap.add_argument(
        "--use_excel_qtype",
        action="store_true",
        help="Read Question_type from Excel per row (mapped to GQ/MCQ/T1/T2/Yes_No)",
    )
    ap.add_argument(
        "--fixed_type",
        default="",
        help="If set (e.g., T1), overrides ALL rows with this type (GQ/MCQ/T1/T2/Yes_No).",
    )

    ap.add_argument("--save_context", action="store_true", help="Warning: JSONL becomes huge")
    ap.add_argument("--force_insufficient_if_bad_cites", action="store_true")
    ap.add_argument("--force_insufficient_if_no_cites", action="store_true")
    ap.add_argument("--debug_first_n", type=int, default=0, help="Print debug for first N rows")

    args = ap.parse_args()

    if not args.openai_api_key and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OpenAI API key. Set OPENAI_API_KEY env var or pass --openai_api_key."
        )

    df = pd.read_excel(args.excel).fillna("")

    # Load vectordb (E5)
    embeddings = HuggingFaceEmbeddings(
        model_name=args.e5_model,
        model_kwargs={"device": "cuda"},  # embeddings on GPU if available
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=args.persist_dir,
    )

    llm = OpenAIResponsesLLM(
        model_name=args.llm_model,
        api_key=args.openai_api_key if args.openai_api_key else None,
        base_url=args.openai_base_url if args.openai_base_url else None,
    )

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixed_qtype = None
    if args.fixed_type and args.fixed_type.strip():
        fixed_qtype = normalize_fixed_type(args.fixed_type)

    system_msg = (
        "You are a grounded QA assistant. Follow the user's Rules strictly. "
        "Do not add extra sections beyond the required format."
    )

    total = len(df)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, r in df.iterrows():
            question = str(r.get(args.col_question, "")).strip()
            options = parse_options_cell(r.get(args.col_options, ""))

            # ---- Question type selection (ONLY your 5 types)
            if fixed_qtype is not None:
                qtype = fixed_qtype
            elif args.use_excel_qtype:
                qtype = normalize_qtype_to_allowed(r.get(args.col_qtype, ""))
            else:
                qtype = "GQ"

            if args.debug_first_n and i < args.debug_first_n:
                print(f"[DEBUG] row={i} excel_qtype='{r.get(args.col_qtype,'')}' -> qtype='{qtype}'")

            # Retrieval query (E5)
            query_used = build_query(question, options)

            kept_docs, kept_ids, raw_count = retrieve_exact_k_unique_docs(
                vectordb=vectordb,
                query=query_used,
                k=args.k,
                fetch_k=args.fetch_k,
            )

            # Context contains the chunk ids shown to the model
            context = build_context_with_ids(kept_docs, kept_ids)
            valid_set = set(kept_ids)

            user_prompt = build_user_prompt(qtype, question, options, context, kept_ids)

            answer = llm.generate_chat(
                system=system_msg,
                user=user_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            cleaned_answer, valid_cites, invalid_cites = sanitize_answer_citations(answer, valid_set)

            if args.force_insufficient_if_bad_cites and invalid_cites:
                cleaned_answer = "Insufficient context."
                valid_cites = []

            if args.force_insufficient_if_no_cites and len(valid_cites) == 0:
                cleaned_answer = "Insufficient context."
                valid_cites = []

            record = {
                "id": int(i),
                "Question_type": qtype,  # ONLY: GQ, MCQ, T1, T2, Yes_No
                "Question": question,
                "Options": options,
                "query_used": query_used,
                "k_retrieval": int(args.k),
                "fetch_k": int(args.fetch_k),

                "retrieved_chunk_ids": kept_ids,
                "num_chunks_retrieved_raw": int(raw_count),
                "num_chunks_retrieved_dedup": int(len(kept_ids)),

                "final_answer": cleaned_answer,
                "final_answer_citations": valid_cites,
                "invalid_citations_removed": invalid_cites,

                "llm_model": args.llm_model,
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
            }

            if args.save_context:
                record["context_used"] = context

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            print(
                f"[{i+1}/{total}] id={i} type={qtype} raw={raw_count} "
                f"kept={len(kept_ids)} bad_cites={len(invalid_cites)} cites={len(valid_cites)}"
            )

    print(f"\n[OK] JSONL saved incrementally at: {out_path}")


if __name__ == "__main__":
    main()

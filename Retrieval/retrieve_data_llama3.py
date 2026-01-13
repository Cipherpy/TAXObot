#!/usr/bin/env python3
"""
FINAL (fixed for LLaMA-3-Instruct): E5 (Chroma) retrieval + HF chat-template generation → incremental JSONL

Fixes:
✅ Uses LLaMA-3 Instruct chat template (apply_chat_template) to stop "assistant.assistant" loops
✅ Token-budgeted context (prevents truncation → fewer "Insufficient context" + less gibberish)
✅ Dedupe to EXACT k unique chunks (by SHIFTED chunk id)
✅ Shifts chunk IDs +1 everywhere (retrieved ids == ids shown to LLM == ids used for citations)
✅ Context shown with explicit chunk ids: [chunk_000123] ...
✅ LLM restricted to cite ONLY valid chunk ids; invalid cites removed
✅ If no citations, automatic retry with stricter instruction (instead of forcing Insufficient)
✅ Output JSONL (incremental append)
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# -----------------------------
# CONFIG DEFAULTS
# -----------------------------
DEFAULT_E5_MODEL = "intfloat/e5-large-v2"
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

    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            out = [str(i).strip() for i in v if str(i).strip()]
            return out or None
    except Exception:
        pass

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
    if fixed_type is None:
        raise ValueError("fixed_type is None")
    return normalize_qtype_to_allowed(fixed_type)


def build_query(question: str, options: Optional[List[str]]) -> str:
    """E5 query prefix required. Options appended if present."""
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
    """Make chunk ids 1-based to match ground truth."""
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
    """Pick a stable chunk id from metadata."""
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
    Returns: kept_docs, kept_shifted_ids, raw_count
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


def build_context_with_ids(docs: List[Any], shifted_ids: List[str]) -> List[str]:
    """
    Return list of chunk-strings:
      ["[chunk_000123] ...", ...]
    We keep list form so we can token-budget safely.
    """
    out = []
    for d, cid in zip(docs, shifted_ids):
        txt = strip_passage_prefix(d.page_content)
        out.append(f"[{cid}] {txt}")
    return out


def extract_citations(text: str) -> List[str]:
    return dedup_preserve_order([m.group(1) for m in CITE_RE.finditer(text or "")])


def sanitize_answer_citations(answer: str, valid_set: set) -> Tuple[str, List[str], List[str]]:
    cites = extract_citations(answer)
    valid = [c for c in cites if c in valid_set]
    invalid = [c for c in cites if c not in valid_set]

    cleaned = answer
    if invalid:
        for c in invalid:
            cleaned = re.sub(rf"\[\s*{re.escape(c)}\s*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[ ]{2,}", " ", cleaned).strip()

    return cleaned.strip(), valid, invalid


# -----------------------------
# LLM (Chat-template for LLaMA-3 Instruct)
# -----------------------------
class HFChatLLM:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # pad_token safety (some llama tokenizers lack it)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()

    def _encode_chat(self, system: str, user: str) -> Dict[str, torch.Tensor]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if self.device == "cuda":
            prompt_ids = prompt_ids.to("cuda")
        return {"input_ids": prompt_ids}

    @torch.no_grad()
    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 4,
    ) -> str:
        enc = self._encode_chat(system, user)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Best-effort: return only the last assistant turn text
        # Many chat templates format as: "...user... assistant ..."
        if "assistant" in text.lower():
            # take last occurrence
            idx = text.lower().rfind("assistant")
            tail = text[idx:]
            # remove the word 'assistant' itself
            tail = re.sub(r"(?i)^assistant[:\s\-]*", "", tail).strip()
            if tail:
                return tail
        return text.strip()

    def count_tokens_chat(self, system: str, user: str) -> int:
        enc = self._encode_chat(system, user)
        return int(enc["input_ids"].shape[-1])


# -----------------------------
# Prompt builder
# -----------------------------
def build_user_prompt(qtype: str, question: str, options: Optional[List[str]], context_chunks: List[str], valid_ids: List[str]) -> str:
    valid_line = ", ".join(valid_ids)

    if qtype == "MCQ":
        fmt = (
            "Return EXACTLY:\n"
            "Answer: <option letter OR exact option text>\n"
            "Explanation: <1-3 lines>\n"
            "Citations: <[chunk_...]>\n"
        )
    elif qtype == "Yes_No":
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

    context_block = "\n\n---\n\n".join(context_chunks)

    return (
        "Use ONLY the provided context.\n"
        "You may ONLY cite chunk IDs that appear in the context.\n"
        f"Valid chunk IDs: {valid_line}\n"
        "Every factual claim must include at least one citation like [chunk_000123].\n"
        "If the answer is not in the context, say exactly: Insufficient context.\n"
        "Do NOT invent citations.\n\n"
        f"{fmt}\n"
        f"Question type: {qtype}\n"
        f"Question: {question.strip()}\n"
        f"{opt_block}\n"
        f"Context:\n{context_block}\n"
    )


def token_budget_context(
    llm: HFChatLLM,
    system: str,
    qtype: str,
    question: str,
    options: Optional[List[str]],
    context_chunks: List[str],
    valid_ids: List[str],
    max_input_tokens: int,
    reserve_output_tokens: int,
) -> List[str]:
    """
    Keep adding chunks until we hit token budget.
    """
    kept: List[str] = []
    for ch in context_chunks:
        trial = kept + [ch]
        user_prompt = build_user_prompt(qtype, question, options, trial, valid_ids)
        toks = llm.count_tokens_chat(system, user_prompt)
        if toks + reserve_output_tokens <= max_input_tokens:
            kept = trial
        else:
            break
    return kept


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

    ap.add_argument("--llm_model", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--col_qtype", default="Question_type")
    ap.add_argument("--col_question", default="Question")
    ap.add_argument("--col_options", default="Options")

    ap.add_argument("--use_excel_qtype", action="store_true")
    ap.add_argument("--fixed_type", default="")

    ap.add_argument("--save_context", action="store_true")
    ap.add_argument("--retry_if_no_cites", action="store_true", help="Retry once if answer has zero valid citations")
    ap.add_argument("--force_insufficient_if_bad_cites", action="store_true")
    ap.add_argument("--debug_first_n", type=int, default=0)

    # token budgeting
    ap.add_argument("--max_input_tokens", type=int, default=8000, help="Model context window (set 8000/16000/128000 etc)")
    ap.add_argument("--reserve_output_tokens", type=int, default=512, help="Reserve for generation")

    args = ap.parse_args()

    df = pd.read_excel(args.excel).fillna("")

    # Load vectordb (E5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=args.e5_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=args.persist_dir,
    )

    llm = HFChatLLM(args.llm_model, device=device)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixed_qtype = None
    if args.fixed_type and args.fixed_type.strip():
        fixed_qtype = normalize_fixed_type(args.fixed_type)

    system_prompt = "You are a grounded QA assistant that follows instructions precisely."

    total = len(df)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, r in df.iterrows():
            question = str(r.get(args.col_question, "")).strip()
            options = parse_options_cell(r.get(args.col_options, ""))

            if fixed_qtype is not None:
                qtype = fixed_qtype
            elif args.use_excel_qtype:
                qtype = normalize_qtype_to_allowed(r.get(args.col_qtype, ""))
            else:
                qtype = "GQ"

            if args.debug_first_n and i < args.debug_first_n:
                print(f"[DEBUG] row={i} excel_qtype='{r.get(args.col_qtype,'')}' -> qtype='{qtype}'")

            query_used = build_query(question, options)

            kept_docs, kept_ids, raw_count = retrieve_exact_k_unique_docs(
                vectordb=vectordb,
                query=query_used,
                k=args.k,
                fetch_k=args.fetch_k,
            )

            # Build chunk strings (with ids) then token-budget them
            context_chunks_all = build_context_with_ids(kept_docs, kept_ids)
            valid_set = set(kept_ids)

            # token-budget: keep as many chunks as fit
            context_chunks = token_budget_context(
                llm=llm,
                system=system_prompt,
                qtype=qtype,
                question=question,
                options=options,
                context_chunks=context_chunks_all,
                valid_ids=kept_ids,
                max_input_tokens=args.max_input_tokens,
                reserve_output_tokens=args.reserve_output_tokens,
            )

            user_prompt = build_user_prompt(qtype, question, options, context_chunks, kept_ids)

            answer = llm.generate(
                system=system_prompt,
                user=user_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            cleaned_answer, valid_cites, invalid_cites = sanitize_answer_citations(answer, valid_set)

            # Optional retry once if zero valid citations (often fixes LLaMA outputs)
            if args.retry_if_no_cites and len(valid_cites) == 0 and "Insufficient context" not in cleaned_answer:
                retry_user = (
                    user_prompt
                    + "\n\nIMPORTANT RETRY:\n"
                      "Rewrite the answer in 1–2 sentences and include 1–2 citations.\n"
                      "Pick citations ONLY from the valid chunk IDs.\n"
                      "Do not repeat words.\n"
                )
                answer2 = llm.generate(
                    system=system_prompt,
                    user=retry_user,
                    max_new_tokens=args.max_new_tokens,
                    temperature=max(args.temperature, 0.2),  # slight sampling to escape loops
                )
                cleaned_answer2, valid_cites2, invalid_cites2 = sanitize_answer_citations(answer2, valid_set)

                # take retry if improved
                if len(valid_cites2) > len(valid_cites):
                    cleaned_answer, valid_cites, invalid_cites = cleaned_answer2, valid_cites2, invalid_cites2

            if args.force_insufficient_if_bad_cites and invalid_cites:
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

                # EXACT ids shown in context
                "retrieved_chunk_ids": kept_ids,
                "num_chunks_retrieved_raw": int(raw_count),
                "num_chunks_retrieved_dedup": int(len(kept_ids)),

                "final_answer": cleaned_answer,
                "final_answer_citations": valid_cites,
                "invalid_citations_removed": invalid_cites,

                "llm_model": args.llm_model,
                "temperature": float(args.temperature),
                "context_chunks_used": int(len(context_chunks)),
            }

            if args.save_context:
                record["context_used"] = "\n\n---\n\n".join(context_chunks)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            print(
                f"[{i+1}/{total}] id={i} type={qtype} raw={raw_count} "
                f"kept={len(kept_ids)} ctx_used={len(context_chunks)} bad_cites={len(invalid_cites)} cites={len(valid_cites)}"
            )

    print(f"\n[OK] JSONL saved incrementally at: {out_path}")


if __name__ == "__main__":
    main()

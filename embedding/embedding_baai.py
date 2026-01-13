# pip install -U langchain langchain-community langchain-huggingface chromadb sentence-transformers

import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# -----------------------------
# CONFIG (BGE-large)
# -----------------------------
BGE_MODEL = "BAAI/bge-large-en-v1.5"


# -----------------------------
# Load chunks JSONL
# -----------------------------
def load_chunks_jsonl(jsonl_path: str) -> List[Dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# -----------------------------
# Metadata sanitizer (Chroma-safe)
# -----------------------------
def sanitize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for k, v in md.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            safe[k] = v
        elif isinstance(v, list):
            if k == "paragraph_span" and len(v) == 2 and all(isinstance(x, int) for x in v):
                safe["para_start"] = v[0]
                safe["para_end"] = v[1]
            else:
                safe[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            safe[k] = json.dumps(v, ensure_ascii=False)
        else:
            safe[k] = str(v)
    return safe


# -----------------------------
# Chunks -> Documents (NO prefix for BGE)
# -----------------------------
def chunks_to_langchain_docs(chunks: List[Dict]) -> List[Document]:
    docs = []
    for i, ch in enumerate(chunks):
        text = (ch.get("text") or "").strip()
        if not text:
            continue

        md = {k: v for k, v in ch.items() if k != "text"}
        md.setdefault("chunk_id", f"chunk_{i:06d}")
        md = sanitize_metadata(md)

        docs.append(Document(page_content=text, metadata=md))
    return docs


# -----------------------------
# Build & persist Chroma
# -----------------------------
def build_chroma_from_chunks(
    jsonl_path: str,
    persist_dir: str = "./chroma_taxobot_bge_large",
    collection_name: str = "taxobot_chunks_bge_large",
    hf_model_name: str = BGE_MODEL,
    device: str = None,  # auto
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    chunks = load_chunks_jsonl(jsonl_path)
    docs = chunks_to_langchain_docs(chunks)

    print(f"Loaded chunks: {len(chunks)} | Documents to embed: {len(docs)}")
    print(f"Embedding model: {hf_model_name} | device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=hf_model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            # optional:
            # "batch_size": 16,
        },
    )

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    vectordb.persist()
    print(f"Chroma DB saved at: {Path(persist_dir).resolve()}")
    return vectordb


if __name__ == "__main__":
    CHUNKS_JSONL = "/home/reshma/TAXObot/Chunking/merged_taxonomic_general_chunks.jsonl"

    db = build_chroma_from_chunks(
        jsonl_path=CHUNKS_JSONL,
        persist_dir="/home/reshma/TAXObot/embedding/chroma_taxobot_bge_large",
        collection_name="taxobot_chunks_bge_large",
        hf_model_name=BGE_MODEL,
        device=None,  # auto cuda/cpu
    )

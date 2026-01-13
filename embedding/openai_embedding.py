# OpenAI embedding version (for reviewer comparison)

import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

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
            # Special-case: paragraph_span = [start, end]
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
# Chunks -> Documents
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
# Build & persist Chroma (OpenAI)
# -----------------------------
def build_chroma_from_chunks(
    jsonl_path: str,
    persist_dir: str = "./chroma_taxobot_openai",
    collection_name: str = "taxobot_chunks_openai",
    openai_model: str = "text-embedding-3-large",  # or text-embedding-3-small
):
    chunks = load_chunks_jsonl(jsonl_path)
    docs = chunks_to_langchain_docs(chunks)

    print(f"Loaded chunks: {len(chunks)} | Documents to embed: {len(docs)}")
    print(f"OpenAI embedding model: {openai_model}")

    embeddings = OpenAIEmbeddings(
        model=openai_model,
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
        persist_dir="/home/reshma/TAXObot/embedding/chroma_taxobot_openai",
        collection_name="taxobot_chunks_openai",
        openai_model="text-embedding-3-large",  # recommended
    )

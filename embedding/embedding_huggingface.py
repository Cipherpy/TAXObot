# pip install -U langchain langchain-community langchain-huggingface chromadb sentence-transformers

import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_chunks_jsonl(jsonl_path: str) -> List[Dict]:
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def sanitize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make metadata Chroma-safe:
    - Convert lists/dicts to strings or split common list fields
    - Keep only primitives: str/int/float/bool/None
    """
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
            # Fallback: stringify anything unknown
            safe[k] = str(v)

    return safe


def chunks_to_langchain_docs(chunks: List[Dict]) -> List[Document]:
    docs = []
    for i, ch in enumerate(chunks):
        text = (ch.get("text") or "").strip()
        if not text:
            continue

        # Separate metadata from text
        md = {k: v for k, v in ch.items() if k != "text"}

        # Ensure stable id
        md.setdefault("chunk_id", f"chunk_{i:06d}")

        md = sanitize_metadata(md)
        docs.append(Document(page_content=text, metadata=md))

    return docs


def build_chroma_from_chunks(
    jsonl_path: str,
    persist_dir: str = "./chroma_taxobot",
    collection_name: str = "taxobot_chunks",
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",  # set "cuda" if you have GPU torch working
):
    chunks = load_chunks_jsonl(jsonl_path)
    docs = chunks_to_langchain_docs(chunks)

    print(f"Loaded chunks: {len(chunks)} | Documents to embed: {len(docs)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=hf_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
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
        persist_dir="./chroma_taxobot",
        collection_name="taxobot_chunks",
        hf_model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda",  # change to "cpu" if cuda not available
    )

   

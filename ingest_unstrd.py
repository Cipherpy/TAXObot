import os
from pathlib import Path
import torch
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document



# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = r"/home/reshma/TAXObot/un_structured"  # folder with PDFs
CHROMA_PERSIST_DIR = r"/home/reshma/TAXObot/un_structured/vectorstore/chroma_unstructured_e5"
COLLECTION_NAME = "taxobot_unstructured_e5"

E5_MODEL = "intfloat/e5-large-v2"
PASSAGE_PREFIX = "passage: "   # for documents/chunks (corpus)
# For queries later, you should use: "query: " prefix


load_dotenv()


def create_vector_db():
    # 1) Load PDFs
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        show_progress=True,
        use_multithreading=True,
    )
    documents = loader.load()
    print(f"Loaded PDFs: {len(documents)}")

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1750,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print("Number of chunks:", len(chunks))

    # 3) Convert to Documents with E5 passage prefix
    docs: list[Document] = []
    for i, ch in enumerate(chunks):
        text = (ch.page_content or "").strip()
        if not text:
            continue

        md = dict(ch.metadata) if ch.metadata else {}
        md.setdefault("chunk_id", f"chunk_{i:06d}")

        docs.append(Document(page_content=PASSAGE_PREFIX + text, metadata=md))

    print(f"Documents to embed: {len(docs)}")

    # 4) Embeddings (E5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=E5_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"Embedding model: {E5_MODEL} | device: {device}")

    # 5) Build + persist Chroma
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    vectordb.persist()

    print(f"Chroma DB saved at: {Path(CHROMA_PERSIST_DIR).resolve()}")
    return vectordb


if __name__ == "__main__":
    create_vector_db()

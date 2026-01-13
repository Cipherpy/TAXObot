

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from core.ui import is_query_valid  # your existing validator


# -----------------------------
# Page + Auth
# -----------------------------
st.set_page_config(layout="wide", page_title="TAXObot")

# Prefer Streamlit secrets; fall back to env
if "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets["openai"]:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["OPENAI_API_KEY"]

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it to Streamlit secrets or environment.")
    st.stop()


# -----------------------------
# CONSTANT CONFIG (NO SIDEBAR)
# -----------------------------
DEFAULT_E5_MODEL = "intfloat/e5-large-v2"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

PERSIST_DIR = "embedding/chroma_taxobot_e5_large"
COLLECTION_NAME = "taxobot_chunks_e5_large"
E5_MODEL = DEFAULT_E5_MODEL
LLM_MODEL = DEFAULT_OPENAI_MODEL

K_RETRIEVAL = 25
FETCH_K = 200

MAX_OUTPUT_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 1.0

FORCE_INSUFFICIENT_IF_BAD_CITES = True
FORCE_INSUFFICIENT_IF_NO_CITES = False

SHOW_DEBUG = False
SHOW_CONTEXT = False


# -----------------------------
# Retrieval / Prompt Config
# -----------------------------
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

CHUNK_PAT = re.compile(r"^(chunk_)(\d+)$", re.IGNORECASE)
CITE_RE = re.compile(r"\[(chunk_\d+)\]", re.IGNORECASE)


# -----------------------------
# Utilities
# -----------------------------
def dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def strip_passage_prefix(text: str) -> str:
    t = (text or "").strip()
    if t.startswith(PASSAGE_PREFIX):
        return t[len(PASSAGE_PREFIX) :].lstrip()
    return t


def build_queryAG_query(question: str) -> str:
    """E5 query prefix required."""
    return QUERY_PREFIX + (question or "").strip()


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

    if isinstance(cid, (int, float)):
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

def build_AG_query(question: str) -> str:
    return QUERY_PREFIX + (question or "").strip()

def retrieve_exact_k_unique_docs(
    vectordb: Chroma, query: str, k: int, fetch_k: int
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

    cleaned = (answer or "").strip()
    if invalid:
        for c in invalid:
            cleaned = re.sub(rf"\[\s*{re.escape(c)}\s*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[ ]{2,}", " ", cleaned).strip()

    return cleaned, valid, invalid


def build_user_prompt(question: str, context: str) -> str:
    return (
        "You are TAXObot, a helpful assistant for marine taxonomy.\n"
        "Use the provided context to answer the user's question.\n"
        "Write naturally and interactively.\n\n"
        "Rules:\n"
        "1) Use ONLY the provided context.\n"
        "2) Do NOT include citations, chunk ids, or bracket references.\n"
        "3) Do NOT use labels like 'Answer:' or 'Citations:'.\n"
        "4) If the answer is not present in the context, say: Insufficient context.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Context:\n{context}\n"
    )



# -----------------------------
# OpenAI Responses API wrapper
# -----------------------------
class OpenAIResponsesLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate_chat(
        self,
        system: str,
        user: str,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        resp = self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        return (resp.output_text or "").strip()


# -----------------------------
# Load Vector DB (Chroma + E5)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_vectordb(persist_dir: str, collection: str, e5_model: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=e5_model,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


@st.cache_resource(show_spinner=False)
def load_llm(model_name: str) -> OpenAIResponsesLLM:
    return OpenAIResponsesLLM(model_name=model_name)


# -----------------------------
# Session State
# -----------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []  # dict per turn
if "generated" not in st.session_state:
    st.session_state["generated"] = ["Hello! Ask me anything about marine organisms and taxonomy 🤗"]
if "past" not in st.session_state:
    st.session_state["past"] = ["Hey! 👋"]


# -----------------------------
# App Title
# -----------------------------
st.title("TAXObot")


# -----------------------------
# Load resources (CONSTANTS)
# -----------------------------
vectordb = load_vectordb(
    persist_dir=PERSIST_DIR,
    collection=COLLECTION_NAME,
    e5_model=E5_MODEL,
)
llm = load_llm(model_name=LLM_MODEL)

SYSTEM_MSG = (
    "You are a grounded QA assistant. Follow the user's Rules strictly. "
    "Do not add extra sections beyond the required format."
)


# -----------------------------
# Layout (2 columns) + CSS + left description
# -----------------------------
col1, col2 = st.columns([1, 3])

st.markdown(
    """
    <style>
    .col1-container {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
    }
    .custom-message .message-avatar img {
        width: 80px;
        height: 80px;
    }
    .custom-message {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .message-avatar {
        margin-right: 10px;
    }
    .message-content {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        max-width: 45%;
    }
    .user-message {
        justify-content: flex-end;
    }
    .bot-message {
        justify-content: flex-start;
    }
    .user-message .message-avatar {
        order: 2;
        margin-left: 10px;
        margin-right: 0;
    }
    .user-message .message-content {
        order: 1;
    }
    .bot-message .message-avatar {
        order: 1;
        margin-left: 10px;
        margin-right: 0;
    }
    .bot-message .message-content {
        order: 2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with col1:
    st.markdown(
        """
        <div class="col1-container">
            <h1></h1>
            <p style='justify-content: center'>
            TAXObot is an AI assistant designed for Marine Taxonomists! In its initial phase, it has assimilated the taxonomic keys of Glyceridae,
            Polychaeta, found in Indian waters. Additional taxonomic keys for various species and groups will be incorporated soon.
            This resource facilitates the identification of organisms by providing their taxonomic characters or vice versa. Furthermore,
            it encompasses general information about Polychaeta. Please feel free to submit any taxonomic inquiries pertaining to Glyceridae in Indian waters
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    response_container = st.container()
    input_container = st.container()

    with input_container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Send a question (:", key="input")
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            if not is_query_valid(user_input):
                st.stop()

            query_used = build_AG_query(user_input)

            kept_docs, kept_ids, raw_count = retrieve_exact_k_unique_docs(
                vectordb=vectordb,
                query=query_used,
                k=K_RETRIEVAL,
                fetch_k=FETCH_K,
            )

            context = build_context_with_ids(kept_docs, kept_ids)
            valid_set = set(kept_ids)

            user_prompt = build_user_prompt(
                question=user_input,
                context=context,
               # valid_ids=kept_ids,
            )

            answer = llm.generate_chat(
                system=SYSTEM_MSG,
                user=user_prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

            # cleaned_answer, valid_cites, invalid_cites = sanitize_answer_citations(answer, valid_set)

            # if FORCE_INSUFFICIENT_IF_BAD_CITES and invalid_cites:
            #     cleaned_answer = "Insufficient context."
            #     valid_cites = []

            # if FORCE_INSUFFICIENT_IF_NO_CITES and len(valid_cites) == 0:
            #     cleaned_answer = "Insufficient context."
            #     valid_cites = []
            cleaned_answer = answer.strip()


            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(cleaned_answer)
            st.session_state["history"].append(
                {
                    "question": user_input,
                    "query_used": query_used,
                    "k": K_RETRIEVAL,
                    "fetch_k": FETCH_K,
                    "num_chunks_retrieved_raw": int(raw_count),
                    "retrieved_chunk_ids": kept_ids,
                    #"valid_citations": valid_cites,
                    #"invalid_citations_removed": invalid_cites,
                    "context_used": context if SHOW_CONTEXT else None,
                }
            )

    if st.session_state["generated"]:
        with response_container:
            for i in range(len(st.session_state["generated"])):
                user_message_html = f'''
                <div class="custom-message user-message">
                    <div class="message-avatar">
                        <img src="https://raw.githubusercontent.com/Cipherpy/TAXObot/main/chat_avatar.png">
                    </div>
                    <div class="message-content">{st.session_state["past"][i]}</div>
                </div>
                '''
                st.markdown(user_message_html, unsafe_allow_html=True)

                bot_message_html = f'''
                <div class="custom-message bot-message">
                    <div class="message-content">{st.session_state["generated"][i]}</div>
                    <div class="message-avatar">
                        <img src="https://raw.githubusercontent.com/Cipherpy/TAXObot/main/icon_polychaete.png">
                    </div>
                </div>
                '''
                st.markdown(bot_message_html, unsafe_allow_html=True)

            if st.session_state["history"] and (SHOW_DEBUG or SHOW_CONTEXT):
                last = st.session_state["history"][-1]

                with st.expander("Retrieval & citation debug", expanded=SHOW_DEBUG):
                    st.write(
                        {
                            "query_used": last["query_used"],
                            "k": last["k"],
                            "fetch_k": last["fetch_k"],
                            "num_chunks_retrieved_raw": last["num_chunks_retrieved_raw"],
                            "num_chunks_retrieved_dedup": len(last["retrieved_chunk_ids"]),
                            "retrieved_chunk_ids": last["retrieved_chunk_ids"],
                            "valid_citations": last["valid_citations"],
                            "invalid_citations_removed": last["invalid_citations_removed"],
                        }
                    )

                if SHOW_CONTEXT and last.get("context_used"):
                    with st.expander("Context used (with chunk ids)", expanded=False):
                        st.text(last["context_used"])

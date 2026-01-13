# streamlit_openai_chroma.py
# Streamlit TAXObot (OpenAI embeddings + Chroma) + GPT-4o (ChatOpenAI)
# - NO sidebar settings (all constants)
# - Interactive answer (answer + brief explanation)
# - NO citations / chunk ids / "Answer:" labels

import os
import re
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from core.ui import is_query_valid  # your existing validator

# -----------------------------
# Page + Auth
# -----------------------------
st.set_page_config(layout="wide", page_title="TAXObot")

# Prefer Streamlit secrets; fall back to env
if "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets["openai"]:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["OPENAI_API_KEY"]

# if not OPENAI_API_KEY:
#     st.error("Missing OPENAI_API_KEY. Add it to environment or .env")
#     st.stop()

# -----------------------------
# CONSTANT CONFIG (NO SIDEBAR)
# -----------------------------
PERSIST_DIR = "embedding/chroma_taxobot_openai"
COLLECTION_NAME = "taxobot_chunks_openai"

OPENAI_EMBED_MODEL = "text-embedding-3-large"
OPENAI_CHAT_MODEL = "gpt-4o-mini"

K_RETRIEVAL = 25

# -----------------------------
# Load VectorDB + LLM
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_vectordb() -> Chroma:
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

@st.cache_resource(show_spinner=False)
def load_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        temperature=0.0,
        max_retries=3,
    )

vectordb = load_vectordb()
llm = load_llm()

# Optional: sanity check (shows doc count)
try:
    n_docs = vectordb._collection.count()
    #st.caption(f"✅ Knowledge base loaded ({n_docs} chunks)")
except Exception:
    st.caption("✅ Knowledge base loaded")

# -----------------------------
# Retrieval
# -----------------------------
def retrieve_contexts(query: str, k: int) -> List[str]:
    docs = vectordb.similarity_search(query, k=k)
    return [d.page_content for d in docs]

# -----------------------------
# Prompt (Interactive, no labels/citations)
# -----------------------------
def build_messages(question: str, contexts: List[str]):
    context_block = "\n\n---\n\n".join(contexts)

    return [
        ("system",
         "You are TAXObot, a marine taxonomy assistant.\n"
         "Answer ONLY using the provided context.\n"
         "Write naturally and interactively.\n"
         "Do NOT include citations, chunk ids, or bracket references.\n"
         "Do NOT use labels like 'Answer:' or 'Explanation:'.\n"
         "If the answer is not present in the context, say exactly:\n"
         '"Not found in the provided document."'),
        ("user",
         f"""Context:
{context_block}

Question: {question}

Respond with a short, direct answer followed by a brief explanation (2–4 lines).""")
    ]

def generate_text(messages, max_tokens: int = 300) -> str:
    resp = llm.bind(max_tokens=max_tokens).invoke(messages)
    return (resp.content or "").strip()

def rag_answer(question: str, k: int = K_RETRIEVAL) -> str:
    contexts = retrieve_contexts(question, k)
    messages = build_messages(question, contexts)
    return generate_text(messages, max_tokens=300)

# -----------------------------
# Session State
# -----------------------------
if "generated" not in st.session_state:
    st.session_state["generated"] = [
        "Hi! I’m TAXObot 🤗 Ask me about marine species, taxonomy, and identification across diverse groups."
    ]
if "past" not in st.session_state:
    st.session_state["past"] = ["Hey! 👋"]

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
        max-width: 55%;
        white-space: pre-wrap;
    }
    .user-message { justify-content: flex-end; }
    .bot-message { justify-content: flex-start; }
    .user-message .message-avatar { order: 2; margin-left: 10px; margin-right: 0; }
    .user-message .message-content { order: 1; }
    .bot-message .message-avatar { order: 2; margin-left: 10px; margin-right: 0; }
    .bot-message .message-content { order: 1; }
    </style>
    """,
    unsafe_allow_html=True,
)

with col1:
    st.markdown(
        '''
        <div class="col1-container" style="margin-bottom: 22px;">
            <p>
                TAXObot is an AI assistant designed for Marine Taxonomists. It supports queries across multiple marine taxonomic groups
                and helps with identification, diagnostic characters, and general biological information derived from the embedded reference documents.
            </p>
        </div>
)

        ''',
        unsafe_allow_html=True
    )

   st.warning(
    "TAXObot does not provide exhaustive coverage of all marine species.\n\n"
    "The current version is limited to a selected subset of species within the listed taxonomic groups.\n\n"
    "This application is a prototype developed for research and evaluation purposes.\n\n"
    "Additional species, taxonomic groups, and enhanced functionalities will be incorporated in future releases."
)





with col2:
    st.title("TAXObot")

    response_container = st.container()
    input_container = st.container()

    with input_container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Send a question (:", key="input")
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            if not is_query_valid(user_input):
                st.stop()

            output = rag_answer(user_input, k=K_RETRIEVAL)

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

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

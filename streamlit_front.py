from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import os
from core.ui import is_query_valid

# Set page configuration
st.set_page_config(layout="wide")

os.environ["OPENAI_API_KEY"] = st.secrets['openai']["OPENAI_API_KEY"]
# load_dotenv()

DB_FAISS_PATH = 'vectorstore/db_faiss_text'

embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings)
vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-4'),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
    return_source_documents=True
)

def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask me anything about Glyceridae 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

# Title
st.title("TAXObot")

# Layout with two columns
col1, col2 = st.columns([1, 3])

# CSS to style the col1 container
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
        background-color: #e0e0e0;
        padding: 10px;
        border-radius: 5px;
    }
    .user-message {
        justify-content: flex-end;
    }
    .bot-message {
        justify-content: flex-start;
    }
    .bot-message .message-avatar {
        order: 2;
        margin-left: 10px;
        margin-right: 0;
    }
    .bot-message .message-content {
        order: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with col1:
    st.markdown(
        """
        <div class="col1-container">
            <h1></h1>
            <p style='justify-content: center'>TAXObot is an AI assistant designed for Marine Taxonomists! In its initial phase, it has assimilated the taxonomic keys of Glyceridae, 
            Polychaeta, found in Indian waters. Additional taxonomic keys for various species and groups will be incorporated soon. 
            This resource facilitates the identification of organisms by providing their taxonomic characters or vice versa. Furthermore, 
            it encompasses general information about Polychaeta. Please feel free to submit any taxonomic inquiries pertaining to Glyceridae in Indian waters</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    # Container for the chat history
    response_container = st.container()
    # Container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Send a question (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            if not is_query_valid(user_input):
                st.stop()
            output = conversational_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
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

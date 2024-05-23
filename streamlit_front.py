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

DB_FAISS_PATH = 'vectorstore/db_faiss_text'

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings)

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
    </style>
    """,
    unsafe_allow_html=True
)

with col1:
    st.markdown(
        """
        <div class="col1-container">
            <h1></h1>
            <p>This AI system is designed to answer your questions about Glyceridae. Feel free to ask anything!</p>
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
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', logo="https://raw.githubusercontent.com/Cipherpy/yolo_objectdetection/main/answer.png")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
# with st.chat_message('assistant', avatar='https://raw.githubusercontent.com/dataprofessor/streamlit-chat-avatar/master/bot-icon.png'):
#   st.write('Hello world!')
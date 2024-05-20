from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import pdfminer.high_level


from core.ui import (
    is_query_valid,
)

load_dotenv()

DB_FAISS_PATH = 'vectorstore/db_faiss_text'

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings)


chain = ConversationalRetrievalChain.from_llm(
llm = ChatOpenAI(temperature=0.0,model_name='gpt-4'),
retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),return_source_documents=True)


def conversational_chat(query):
        
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        # print(result)
        return result["answer"]
    
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about " + "Polychaete" + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]
        
#container for the chat history
response_container = st.container()
#container for the user's text input
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
    print(st.session_state['generated'])
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

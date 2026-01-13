
import streamlit as st
# from langchain.docstore.document import Document

import openai
from streamlit.logger import get_logger
from typing import NoReturn


def is_query_valid(query: str) -> bool:
    if not query:
        st.error("Please enter a question!")
        return False
    return True

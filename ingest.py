from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

DATA_PATH = "D:/CMLRE/LanguageModel/Paper_Sahu/text/"
DB_FAISS_PATH = 'vectorstore/db_faiss_text'
os.environ["OPENAI_API_KEY"] = st.secrets['openai']["OPENAI_API_KEY"]
# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.txt')

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n"],chunk_size=2000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print("Number of chunks:", len(texts))
    for index, chunk in enumerate(texts):
        print(f"Chunk{index+1}:")
        print(chunk)
        print("---")
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
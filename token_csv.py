import re
import csv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from dotenv import load_dotenv


DATA_PATH = "D:/CMLRE/LanguageModel/TAXOBOT/semi_structured/"
DB_FAISS_PATH = 'vectorstore/db_faiss_semi'
load_dotenv()

# Function to get tokens from a text chunk
def get_tokens(text):
    return re.findall(r"\b\w+\b", text)

# Find the maximum number of tokens in any chunk
def find_max_tokens(texts):
    max_tokens = 0
    for text_list in texts:
        text=text_list.page_content
        
        max_tokens = max(max_tokens, len(get_tokens(text)))
    return max_tokens

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.txt')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n","\n"," ",""],chunk_size=1750, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    max_tokens = find_max_tokens(texts)

    # Open a CSV file to write the chunk numbers and tokens
    with open("output_tokens.csv", "w", newline='', encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        # Write header
        header = ["Chunk Number"] + [f"Token {i+1}" for i in range(max_tokens)]
        csv_writer.writerow(header)

        if texts and isinstance(texts[0], list):
            # Case for multiple documents
            for doc_index, chunks in enumerate(texts):
                for i, chunk in enumerate(chunks):
                    tokens = get_tokens(chunk)
                    row = [f"Doc {doc_index + 1} Chunk {i + 1}"] + tokens + [''] * (max_tokens - len(tokens))
                    csv_writer.writerow(row)
        else:
            # Case for a single document
            for i, chunk in enumerate(texts):
                chunk=str(chunk.page_content)
                tokens = get_tokens(chunk)
                row = [f"Chunk {i + 1}"] + tokens + [''] * (max_tokens - len(tokens))
                csv_writer.writerow(row)

# Main execution
if __name__ == "__main__":
    create_vector_db()

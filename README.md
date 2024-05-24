# TAXObot
TAXObot is an AI assistant designed for Marine Taxonomists, particularly focusing on the taxonomic information of Glyceridae, a family of Polychaeta found in Indian waters. This Streamlit application leverages Retrieval-Augmented Generation (RAG) to provide accurate and detailed taxonomic information based on user queries.






## Features

* Conversational AI: Interact with TAXObot using natural language to get detailed taxonomic information.
* RAG Implementation: Combines language generation and retrieval for precise and contextually accurate answers.
* User-Friendly Interface: Clean and intuitive interface built with Streamlit.



## Demo
https://github.com/Cipherpy/TAXObot/assets/27478550/d6981001-8ad8-4e26-853b-843bc9052957

## Installation
### 1. Clone the repository

    git clone https://github.com/Cipherpy/TAXObot.git 
    cd TAXObot
    
### 2. Create and activate a virtual environment (https://www.tensorflow.org/install/pip)
        
        conda create --name TAXObot python=3.9
        conda activate TAXObot
### 3. Install the dependencies

        pip install -r requirements.txt

### 4. Set up environment variables
- Create a `.env` file in the root directory of the project.
- Add your OpenAI API key in the `.env` file

          OPENAI_API_KEY=your_openai_api_key

### 5. Run the Streamlit app

        streamlit run streamlit_front.py

## Usage
* Once the app is running, open your web browser and go to http://localhost:8501.
* Start interacting with TAXObot by typing your queries related to marine polychaetes in the input box.

## Knowledge Base
The knowledge base is a collection of information that TAXObot uses to answer your questions. It's like a library that contains detailed taxonomic information on Glyceridae. When you ask a question, the RAG model searches through this knowledge base to find relevant information and then uses it to generate a precise and accurate answer.

* **PDF Data**
* 
    - One part of the knowledge base consists of PDF documents that contain extensive taxonomic information on Glyceridae. These PDFs have been processed and indexed to allow the RAG model to retrieve relevant sections based on your queries.
* **TEXT Data**
* 
    - Another part of the knowledge base consists of custom-created text data files. These files contain detailed descriptions and taxonomic keys specifically tailored by our team to enhance the accuracy and relevance of the information provided by TAXObot. This bespoke text data ensures that the model has access to the most precise and up-to-date information

The knowledge bases for the RAG model are stored in the **pdf/** and **text/** folders, respectively.

## Sample Questions and Answers
The **samples/** folder contains a set of sample questions and answers to help you get started with the kinds of queries you can ask TAXObot. Here are a few examples:

1. **Question**: What are the key identifying features of Glyceridae?
   
    - **Answer**: Glyceridae are identified by their elongated, segmented bodies with numerous bristles (chaetae) on each segment. They also possess a distinct head with sensory appendages.
3. **Question**: Where can Glyceridae be found?
   
    - **Answer**: Glyceridae are commonly found in marine environments, particularly in sandy or muddy substrates where they burrow and hunt for prey.

## Project Structure
- streamlit_front.py: Main application script.
- core/: Core functions and utilities.
- vectorstore/: Directory for storing vector database.
- requirements.txt: List of dependencies.
- README.md: This file.
- .env: Environment variables file (not included, must be created manually).

## Contributing
### We welcome contributions to enhance TAXObot. To contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes.
4. Commit your changes (git commit -m 'Add some feature').
5. Push to the branch (git push origin feature-branch).
6. Open a pull request.

## Contact
### For questions or issues, please open an issue in this repository or contact the maintainers:

#### Name: Reshma B, Nosad Sahu
#### Email: reshmababuraj89@gmail.com, nosadsahu@gmail.com

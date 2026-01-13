# TAXObot

TAXObot is an AI assistant designed for Marine Taxonomists, particularly focusing on the taxonomic information of 500 marine species representing multiple taxonomic groups, including Polychaeta, Crustacea, Asteroidea, Holothuroidea, Chondrichthyes, and Pisces. This Streamlit application leverages Retrieval-Augmented Generation (RAG) to provide accurate and detailed taxonomic information based on user queries.

## Features

* **Conversational AI**: Interact with TAXObot using natural language to get detailed taxonomic information.
* **RAG Implementation**: Combines language generation and retrieval for precise and contextually accurate answers.
* **User-Friendly Interface**: Clean and intuitive interface built with Streamlit.



## Demo 
( https://taxobot-citvryrvrvqrrk4t8hrupo.streamlit.app/)

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

        streamlit run streamlit_semi.py

## Usage
* Once the app is running, open your web browser and go to http://localhost:8501.
* Start interacting with TAXObot by typing your queries related to marine polychaetes in the input box.

## Knowledge Base
The knowledge base is a collection of information that TAXObot uses to answer your questions. It's like a library that contains detailed taxonomic information on selected species spanning multiple groups . When you ask a question, the RAG model searches through this knowledge base to find relevant information and then uses it to generate a precise and accurate answer.

* **UNSTRUCTURED Data**
 
    - One part of the knowledge base consists of PDF documents that contain extensive taxonomic information on different species. These PDFs have been processed and indexed to allow the RAG model to retrieve relevant sections based on your queries.
* **SEMI-STRUCTURED Data**
  
    - Another part of the knowledge base consists of custom-created text data files. These files contain detailed descriptions and taxonomic keys specifically tailored by our team to enhance the accuracy and relevance of the information provided by TAXObot. This bespoke text data ensures that the model has access to the most precise and up-to-date information

The knowledge bases for the RAG model are stored in the `un_structured/` and `semi_structured/` folders, respectively.

## Custom chunking
A custom chunking strategy was implemented to distinguish between general descriptive text and taxonomic species descriptions. General content is segmented using header detection based on bold-character ratio, where paragraphs exceeding a predefined boldness threshold are treated as section headers and initiate new chunks. Taxonomic content is chunked using explicit “Species name:” tags, which act as hard anchors defining the start of each species-specific block. All text following a species tag is grouped into a single taxonomic chunk until the next species tag appears, preserving diagnostic continuity.

### Metadata-Enriched Chunk Representation
Each chunk is stored with structured metadata to support filtering, evaluation, and explainability
- `chunk_type`: general / taxonomic.
- `species_name`: present only for taxonomic chunks.
- `section_label`: header or subsection name.
- `source_document`: original reference.
- `chunk_text`: cleaned textual content.

## Data Ingestion
To ingest data into the RAG model, use the data ingestion script. This script processes the data files and indexes them for retrieval by the model.

### Running the Ingest Script
1. Ensure your data is placed in the appropriate folders:

    - Unstructured documents should be placed in `un_structured/`
    - Custom semi structured data files should be placed in `semi_structured/`
2. Run the ingestion script:
    - For the semistructured dataset
   
      ```
      python ingest_semi.py
      ```
      
    - For the unstructured data
      
      ```
      python ingest_unstrd.py
      ```
      
This will process and index the data, making it available for retrieval by TAXObot.

## Sample Questions and Answers

The file `Supplementary_data/Questions.xlsx` contains a curated set of sample questions and corresponding expected answers used for evaluation, testing, and demonstration of TAXObot’s capabilities. These questions reflect the different types of queries that users can pose to the system, including general, taxonomic, and descriptive questions.
Here are a few examples:

1. **Question**: What are the key identifying features of Glyceridae?
   
    - **Answer**: Glyceridae are identified by their elongated, segmented bodies with numerous bristles (chaetae) on each segment. They also possess a distinct head with sensory appendages.
3. **Question**: Where can Glyceridae be found?
   
    - **Answer**: Glyceridae are commonly found in marine environments, particularly in sandy or muddy substrates where they burrow and hunt for prey.

## Project Structure
- `streamlit_semi.py`, `streamlit_unstrd.py`: Main application script.
- `core/`: Core functions and utilities.
- Knowledge bases for the RAG model contained in.
    - `un_structured/`: Contains unstructured documents.
    - `semi_structured/`: Contains custom-created text data files.
- `Sample/`: Contains sample questions and answers.
- `vectorstore/`: Directory for storing vector database.
- `ingest_semi.py`,  `ingest_unstrd.py`: Script to ingest data into the RAG model.
- `requirements.txt`: List of dependencies.
- `README.md`: This file.
- `.env`: Environment variables file (not included, must be created manually).

## ⚠️ Disclaimer

- TAXObot does not provide exhaustive coverage of all marine species.
- The current version is limited to a selected subset of species within the listed taxonomic groups.
- This application is a prototype developed for research and evaluation purposes.
- Additional species, taxonomic groups, and enhanced functionalities will be incorporated in future releases.

## Contributing
### We welcome contributions to enhance TAXObot. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## Contact
For questions or issues, please open an issue in this repository or contact the maintainers

#### Name: Reshma B, Nosad Sahu
#### Email: reshmababuraj89@gmail.com, nosadsahu@gmail.com

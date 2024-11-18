# RAG System for PDF Response Generation

This project implements a Retrieval-Augmented Generation (RAG) system that generates responses based on the content of a PDF file (`loi-n-01-00-portant-organisation-de-lenseignement-supérieur.pdf`). It utilizes Langchain for tokenization, Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model for embedding, and FAISS for storing and retrieving the most similar chunks. Finally, it integrates LLaMA 3 (70B) with the help of the GROQ API to generate responses.

## Project Structure

```
.
├── app/                             # Application code for the RAG system
│   ├── __init__.py                  # Initializes the app module
│   ├── main.py                      # FastAPI application - main entry point for the API
│   ├── rag_with_py.py               # RAG logic and chunk retrieval (PDF processing, embedding, FAISS)
│   ├── loi-n-01-00-portant-organisation-de-lenseignement-supérieur.pdf  # PDF for processing
│   └── static/                      # Static files (e.g., frontend HTML)
│       └── index.html               # Frontend HTML for interacting with the API
│       └── styles.css               
│       └── script.js
├── requirements.txt                 # List of project dependencies
├── config.json                      # Configuration file for API keys and settings (Groq API)
└── tests/                           # Test suite for the project
    ├── __init__.py                  # Initializes the test module
    ├── test_main.py                 # Unit tests for FastAPI application
    └── test_rag_with_py.py          # Unit tests for RAG logic (PDF chunking, embedding, etc.)

```

## Requirements

This project requires Python and several libraries listed in the `requirements.txt` file. To ensure compatibility and ease of setup, it's recommended to use a virtual environment.

## Setup Instructions

### Step 1: Create and Activate a Virtual Environment

1. **Navigate to your project directory**:
   ```bash
   cd /path/to/your/project
   ```

2. **Create a virtual environment** (you can name it `venv` or any name you prefer):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

### Step 2: Install Dependencies

Once the virtual environment is activated, install the required libraries using the following command:

```bash
pip install -r requirements.txt
```
### Step 3: Set Up Configuration

Create a config.json file in your project directory with the following structure:
```bash
{
    "api_key": "your_groq_api_key_here"
}
```
=> Replace your_groq_api_key_here with your actual Groq API key.

### Step 4: Run the Application
1. **Start the FastAPI server:**
```bash
uvicorn app.main:app --reload
```
2. **Access the application: Open your browser and go to http://127.0.0.1:8000 to interact with the RAG system.**

## Testing
To test the system, unit tests have been created for the main.py (FastAPI) and rag_with_py.py (RAG logic). These tests are located in the tests/ folder. You can run the tests using the following command:
```bash
pytest tests/

```

## Usage
Once the server is running, the system will process the provided PDF (loi-n-01-00-portant-organisation-de-lenseignement-supérieur.pdf), extract the content, generate text embeddings, store them in a FAISS index, and be ready to respond to queries based on the document. The RAG system leverages the LLaMA 3 (70B) model via the GROQ API for generating responses.

## Custom PDF Files
You can easily work with your own PDF files by modifying the path in the code where the PDF is loaded. Just ensure that the PDF file is accessible in your project directory or provide an absolute path to the file.

## Future Enhancementsl
- Improved Information Extraction: Enhance the extraction capabilities to better handle images, tables, and graphs within PDF files, enabling more comprehensive data retrieval and analysis.
- Hallucination Mitigation: Implement strategies to reduce AI hallucinations, improving the accuracy and reliability of the extracted information and summaries generated from PDF content.


## Contributing

If you wish to contribute to this project, please fork the repository and submit a pull request. Any feedback or suggestions for improvements are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.

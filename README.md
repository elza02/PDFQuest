# RAG System for PDF Response Generation

This project implements a Retrieval-Augmented Generation (RAG) system that generates responses based on the content of a PDF file (`loi-n-01-00-portant-organisation-de-lenseignement-supérieur.pdf`). It utilizes Langchain for tokenization, Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model for embedding, and FAISS for storing and retrieving the most similar chunks. Finally, it integrates LLaMA 3 (70B) with the help of the GROQ API to generate responses.

## Project Structure

```
.
├── requirements.txt
├── RAG_fr_scratch.ipynb
└── loi-n-01-00-portant-organisation-de-lenseignement-supérieur.pdf
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

## Usage

Once executed, the system will first convert the content of the provided PDF file to plain text, as Hugging Face embedding models accept only plain text. Next, it will tokenize the plain text, divide it into chunks, and generate embeddings using the Hugging Face model. These embeddings will then be stored in FAISS. Finally, the RAG system will utilize LLaMA 3 through the GROQ API to answer queries based on the content of the PDF.

## Future Enhancements
Improved Information Extraction: Enhance the extraction capabilities to better handle images, tables, and graphs within PDF files, enabling more comprehensive data retrieval and analysis.
Application Interface: Develop a user-friendly application interface to streamline interactions, allowing users to upload, process, and review extracted information efficiently.
Hallucination Mitigation: Implement strategies to reduce AI hallucinations, improving the accuracy and reliability of the extracted information and summaries generated from PDF content.

## Custom PDF Files

You can easily work with your own PDF files by modifying the path in the code where the PDF is loaded. Just ensure that the PDF file is accessible in your project directory or provide an absolute path to the file.

## Contributing

If you wish to contribute to this project, please fork the repository and submit a pull request. Any feedback or suggestions for improvements are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.

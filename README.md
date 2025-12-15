# Secure RAG System

This application demonstrates a RAG (Retrieval Augmented Generation) system that uses prompt engineering to secure sensitive data.

## Features
- **File Upload**: Supports CSV, PDF, TXT.
- **RAG Pipeline**: Uses LangChain and ChromaDB.
- **Security**: Strictly prohibits revealing PII (Account Numbers, Phone Numbers, Salaries, Credit Scores) via System Prompt.
- **Aggregated Insights**: Allows analytical questions about the data.

## Setup
The project dependencies are listed in `requirements.txt`.
You can install them using:
```bash
py -m pip install -r requirements.txt
```

## Running the App
To start the Streamlit application:
```bash
py -m streamlit run app.py
```

## Usage
1.  **Enter API Key**: In the sidebar, enter your Google Gemini API Key (or OpenAI Key).
2.  **Upload Data**: Use the example `mock_bank_statement.csv` provided in this directory.
3.  **Ask Questions**:
    -   *Blocked*: "What is John Doe's account number?"
    -   *Allowed*: "What is the average transaction amount?"

# PDF Querying System Using LangChain and Claude AI

This project is a Python-based application that allows you to upload a directory of PDF documents, create embeddings for their content using OpenAI, store them in a FAISS vector database, and query the documents using Anthropic's Claude AI model.

## Features
- **PDF Processing**: Loads PDF files from a specified directory.
- **Text Splitting**: Splits documents into manageable chunks for embedding.
- **Embeddings with OpenAI**: Uses OpenAI's `text-embedding-3-large` model to generate embeddings.
- **FAISS Integration**: Stores document embeddings in a FAISS vector database.
- **Query System**: Uses Anthropic's Claude AI to answer questions about the documents' content.
- **Custom Prompt**: Fetches and uses a retrieval-augmented generation (RAG) prompt for the query responses.

---

## Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or later
- `langchain`, `langchain_community`, `langchain_openai`, `langchain_anthropic`, `anthropic`
- FAISS library
- OpenAI and Anthropic API keys

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/pdf-querying-system.git
   cd pdf-querying-system

# RAG Application with LangChain, Gemini, FAISS and AWS

This project implements a Retrieval-Augmented Generation (RAG) application that answers questions based on a collection of documents. It leverages a powerful stack of modern AI tools to provide accurate, context-aware answers. The application can source documents (PDFs, text files, Word documents) from either a local directory or an AWS S3 bucket.

## Features

- **Multi-Format Document Support**: Ingests `.pdf`, `.txt`, and `.docx` files.
- **Flexible Data Sources**: Load documents from a local `data` folder or directly from an AWS S3 bucket.
- **State-of-the-Art LLM**: Utilizes Google's Gemini-flash-lite model for question answering.
- **Efficient Retrieval**: Employs FAISS for fast and efficient similarity searches in a vector store.
- **High-Quality Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` embedding model for generating document embeddings.
- **Persistent Vector Store**: Caches the FAISS index locally to avoid reprocessing documents on every run, saving time and computational resources.

## How It Works

1. **Load**: Documents are loaded from the specified source (local or S3).
2. **Split**: The documents are broken down into smaller, manageable chunks.
3. **Embed**: Each chunk is converted into a numerical vector (embedding) using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model.
4. **Store**: These embeddings are stored in a FAISS vector index, which allows for very fast searching of the most relevant document chunks.
5. **Retrieve & Generate**: When you ask a question:
   - Your question is converted into an embedding.
   - FAISS finds the document chunks with embeddings most similar to your question's embedding.
   - These relevant chunks (the context) and your original question are passed to the Gemini LLM.
   - Gemini generates a final answer based on the provided context.

## Setup Instructions

### 1. Install Dependencies

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

### 2. Configure API Keys and AWS Credentials

#### Google API Key

You need a Google API key to use the Gemini model.

1. Create a file named `.env` in the root of the project directory.
2. Add your API key to this file:

```env
GOOGLE_API_KEY="AIzaSy...your...google...api...key"
```

#### AWS Credentials (Optional)

If you plan to use an S3 bucket as your data source, configure your AWS credentials. The most common way is to use the AWS CLI:

```bash
aws configure
```

This will set up the necessary credentials in `~/.aws/credentials`, which `boto3` (the AWS SDK for Python) will automatically detect.

## How to Run

### Step 1: Place Your Documents

- **For Local Mode**: Create a directory named `data` in the project root and place your `.pdf`, `.txt`, and `.docx` files inside it.
- **For S3 Mode**: Upload your documents to your S3 bucket.

### Step 2: Run the Application

The script `rag_app.py` is the entry point.

- **To run with local documents** (default mode, looks for the `data` directory):

```bash
python rag_app.py
```

- **To run with documents from an S3 bucket** (use the `--source s3` and `--s3_bucket` flags):

```bash
python rag_app.py --source s3 --s3_bucket your-s3-bucket-name
```

### Forcing the Index to Rebuild

The first time you run the application, it creates a `faiss_index` directory to store the vector index. On subsequent runs, it loads this pre-built index for faster startup. If you add, remove, or change your source documents, rebuild the index using the `--rebuild_index` flag:

- **For local files**:

```bash
python rag_app.py --rebuild_index
```

- **For S3 files**:

```bash
python rag_app.py --source s3 --s3_bucket your-s3-bucket-name --rebuild_index
```

## Interacting with the Application

Once the application is running and you see the "RAG Application Ready" message, you can start asking questions. Type your question and press Enter. To exit, type `exit` or `quit`.
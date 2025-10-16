import os
import argparse
import logging
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
LOCAL_DATA_DIR = "data"
FAISS_INDEX_PATH = "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Document Loading ---

class DocumentLoaderError(Exception):
    """Custom exception for document loading errors."""
    pass

def get_loader(file_path):
    """Returns the appropriate LangChain document loader based on file extension."""
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == ".pdf":
        return PyPDFLoader(file_path)
    elif file_extension.lower() == ".txt":
        return TextLoader(file_path)
    elif file_extension.lower() == ".docx":
        return Docx2txtLoader(file_path)
    else:
        logging.warning(f"Unsupported file type: {file_path}. Skipping.")
        return None

def load_documents_from_directory(directory_path):
    """Loads all supported documents from a given directory."""
    documents = []
    if not os.path.isdir(directory_path):
        raise DocumentLoaderError(f"Directory not found: {directory_path}")
        
    logging.info(f"Loading documents from {directory_path}...")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                loader = get_loader(file_path)
                if loader:
                    documents.extend(loader.load())
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {e}")
    logging.info(f"Loaded {len(documents)} documents.")
    return documents

def download_from_s3(bucket_name, local_dir):
    """Downloads files from an S3 bucket to a local directory."""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    logging.info(f"Attempting to download files from S3 bucket: {bucket_name}")
    try:
        s3 = boto3.client('s3')
        # Check if bucket exists and is accessible
        s3.head_bucket(Bucket=bucket_name)
        
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        
        for page in pages:
            if "Contents" in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    local_file_path = os.path.join(local_dir, os.path.basename(key))
                    if not key.endswith('/'): # Skip directories
                        logging.info(f"Downloading {key} to {local_file_path}")
                        s3.download_file(bucket_name, key, local_file_path)

    except (NoCredentialsError, PartialCredentialsError):
        logging.error("AWS credentials not found. Please configure them.")
        raise
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logging.error(f"S3 bucket '{bucket_name}' not found.")
        elif e.response['Error']['Code'] == '403':
            logging.error(f"Access denied to S3 bucket '{bucket_name}'.")
        else:
            logging.error(f"An S3 client error occurred: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred with S3: {e}")
        raise

# --- Vector Store and RAG Chain ---

def create_vector_store(documents):
    """Creates a FAISS vector store from documents."""
    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(documents)
    
    logging.info("Generating embeddings and creating FAISS index...")
    # Use HuggingFace model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    vector_store = FAISS.from_documents(docs, embeddings)
    
    logging.info(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store

def create_rag_chain(vector_store):
    """Creates the RAG chain for question answering."""
    logging.info("Initializing Gemini LLM and creating RAG chain...")
    
    # Initialize the Gemini model
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini LLM. Check your GOOGLE_API_KEY. Error: {e}")
        raise

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Keep the answer concise and helpful.

    Context: {context}

    Question: {input}

    Answer:
    """)

    # Create the main chain components
    retriever = vector_store.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the final retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    logging.info("RAG chain created successfully.")
    return retrieval_chain

def main():
    """Main function to run the RAG application."""
    parser = argparse.ArgumentParser(description="RAG Application using LangChain and Gemini")
    parser.add_argument(
        "--source", 
        type=str, 
        default="local", 
        choices=["local", "s3"],
        help="The source of the documents (local directory or AWS S3 bucket)."
    )
    parser.add_argument(
        "--s3_bucket", 
        type=str, 
        help="The name of the S3 bucket (required if source is 's3')."
    )
    parser.add_argument(
        "--rebuild_index",
        action="store_true",
        help="Force rebuilding the FAISS index even if it exists."
    )
    args = parser.parse_args()

    # --- API Key Check ---
    if not os.getenv("GOOGLE_API_KEY"):
        logging.error("GOOGLE_API_KEY environment variable not set.")
        print("\nPlease create a .env file and add your key: GOOGLE_API_KEY='your-api-key'")
        return

    vector_store = None
    
    # --- Load or Create Vector Store ---
    if not args.rebuild_index and os.path.exists(FAISS_INDEX_PATH):
        try:
            logging.info(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
            embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            logging.info("Index loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load FAISS index: {e}. Will try to rebuild.")
            args.rebuild_index = True # Force rebuild if loading fails

    if args.rebuild_index or vector_store is None:
        try:
            documents = []
            if args.source == "local":
                documents = load_documents_from_directory(LOCAL_DATA_DIR)
            elif args.source == "s3":
                if not args.s3_bucket:
                    logging.error("S3 bucket name must be provided with --s3_bucket")
                    return
                download_from_s3(args.s3_bucket, LOCAL_DATA_DIR)
                documents = load_documents_from_directory(LOCAL_DATA_DIR)
            
            if not documents:
                logging.error("No documents were loaded. Cannot build index. Exiting.")
                return

            vector_store = create_vector_store(documents)
        except Exception as e:
            logging.error(f"An error occurred during index creation: {e}")
            return
            
    # --- Create RAG Chain and start conversation ---
    try:
        rag_chain = create_rag_chain(vector_store)
        
        print("\n--- RAG Application Ready ---")
        print("Ask questions about your documents. Type 'exit' or 'quit' to stop.")
        while True:
            query = input("\nYour question: ")
            if query.lower() in ["exit", "quit"]:
                break
            if not query.strip():
                continue
                
            logging.info(f"Invoking chain with query: '{query}'")
            response = rag_chain.invoke({"input": query})
            
            print("\nAnswer:")
            print(response["answer"])

            # Extract and display sources from the context
            if response.get("context"):
                sources = {doc.metadata.get("source", "Unknown") for doc in response["context"]}
                if sources:
                    print("\nSources:")
                    # Sort sources for consistent output and show only the filename
                    for source in sorted(list(sources)):
                        print(f"- {os.path.basename(source)}")

    except Exception as e:
        logging.error(f"An error occurred while running the conversation chain: {e}")

if __name__ == "__main__":
    main()
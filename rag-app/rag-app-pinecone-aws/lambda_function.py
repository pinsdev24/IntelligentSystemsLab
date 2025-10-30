import json
import boto3
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
import os
import tempfile
from pinecone import Pinecone

s3 = boto3.client('s3')

# configuration
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
BUCKET_NAME = os.environ['BUCKET_NAME']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Cache of vector store
vector_store = None

def load_documents_from_s3(bucket_name):
    documents = []
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        for item in response.get('Contents', []):
            if item['Key'].endswith(('.pdf', '.txt')):
                # Download the file from S3
                s3_response = s3.get_object(Bucket=bucket_name, Key=item['Key'])
                file_content = s3_response['Body'].read()
                
                # Create a temporary file to store the downloaded content
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                # Load the document using the appropriate loader
                if item['Key'].endswith('.pdf'):
                    loader = PyPDFLoader(temp_file_path)
                else:
                    loader = TextLoader(temp_file_path)
                
                documents.extend(loader.load())
                
                # Clean up the temporary file
                os.remove(temp_file_path)
    except Exception as e:
        print(f"Error loading documents from S3: {e}")
    return documents

def lambda_handler(event, context):
    global vector_store
    
    if vector_store is None:
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        documents = load_documents_from_s3(BUCKET_NAME)
        
        # If documents are found, build/update the vector store
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            splits = text_splitter.split_documents(documents)
            vector_store = PineconeVectorStore.from_documents(splits, embeddings, index_name=PINECONE_INDEX_NAME)
        # If no documents, try to load from existing index
        else:
            try:
                vector_store = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)
            except Exception as e:
                print(f"Could not load from existing index, and no documents to load: {e}")
                vector_store = None

    if vector_store is None:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Could not initialize vector store. No documents found and existing index could not be loaded.'})
        }
    
    # Retrive the query from the event
    body = json.loads(event['body'])
    query = body.get('query')
    
    if not query:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Query parameter is required'})
        }
    
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    
    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    # Run the query through the chain
    result = qa_chain.invoke({"query": query})
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps({
            'results': result['result'], 
            'source_documents': [doc.page_content for doc in result['source_documents']],
            'sources': [
                {
                'content': doc.page_content[:100], 
                'source': doc.metadata.get('source')
                } for doc in result['source_documents']
            ]
        })
    }

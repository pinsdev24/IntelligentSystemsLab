# RAG with Pinecone, Google Embeddings and AWS

This project implements a serverless RAG application on AWS Lambda, with document storage in S3, embeddings generated via the Google API (`text-embedding-004`), and vector storage in Pinecone. Requests are exposed via API Gateway.

## Prerequisites
- Python 3.8+
- API Keys: Google (for Gemini and embeddings) and Pinecone.
- AWS CLI and SAM CLI.
- Dependencies: See `requirements.txt`.

## Installation
1. Configure an S3 bucket:
   ```bash
   aws s3 mb s3://your-rag-bucket
   aws s3 cp data/ s3://your-rag-bucket/ --recursive
   ```
2. Create a Pinecone index (`rag-index`, dimension 768, cosine metric).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt -t package/
   ```
4. Deploy with SAM:
   ```bash
   sam build
   sam deploy --guided
   ```

## Usage
Send a POST request to the API:
```bash
curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/Prod/query \
-H "Content-Type: application/json" \
-d '{"query": "Summarize the main content."}'
```

## Structure
- `lambda_function.py`: Main Lambda function.
- `template.yaml`: SAM template for deployment.
- `requirements.txt`: Python dependencies.

## Notes
- Google embeddings reduce the package size by eliminating `sentence-transformers`.
- For native AWS integration, consider Amazon Bedrock instead of Gemini.
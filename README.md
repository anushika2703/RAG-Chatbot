# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using Python, LangGraph, Pinecone, Hugging Face embeddings, Groq, and FastAPI.

The chatbot answers questions strictly using the provided Agentic AI ebook as its knowledge base. If the requested information cannot be found in the retrieved context, the system returns a grounded fallback response instead of generating an answer from outside knowledge.

---

## Features

- PDF ingestion and text extraction
- Recursive text chunking with overlap
- Semantic embeddings using `all-MiniLM-L6-v2`
- Vector storage and similarity search using Pinecone
- LangGraph-based retrieval and generation workflow
- Groq LLM for answer generation
- Strict context-grounded prompting
- Relevance-based confidence score
- Retrieved context returned with every response
- Source document and page information
- FastAPI REST API
- Out-of-domain question fallback

---

## Architecture

```text
                 Agentic AI PDF
                       |
                       v
                PDF Text Loader
                       |
                       v
                  Chunking
                       |
                       v
              Hugging Face Embeddings
                       |
                       v
                    Pinecone
                       |
                       |
User Question --> FastAPI
                       |
                       v
                   LangGraph
                       |
                Retrieve Top-K
                       |
                       v
               Relevance Check
                       |
                 +-----+-----+
                 |           |
              Relevant    Not Relevant
                 |           |
                 v           v
             Groq LLM    Fallback
                 |
                 v
        Answer + Context + Score
                 |
                 v
               API Response


Tech Stack
Python
FastAPI
LangGraph
LangChain
Pinecone
Hugging Face Sentence Transformers
Groq
Pydantic
Uvicorn
Project Structure
app/
├── ingestion/
│   └── ingest.py
├── rag/
│   ├── embeddings.py
│   ├── graph.py
│   ├── prompts.py
│   └── retriever.py
├── config.py
├── main.py
└── schemas.py

data/
└── agentic_ai.pdf
Setup
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG-Chatbot
2. Create a virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

Create a .env file in the project root:

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag
GROQ_API_KEY=your_groq_api_key

Do not commit .env to GitHub.

PDF Ingestion

Place the Agentic AI ebook at:

data/agentic_ai.pdf

Run:

python -m app.ingestion.ingest

This performs:

PDF loading
Text extraction
Chunking
Embedding generation
Pinecone indexing
Run the API

Start the FastAPI server:

uvicorn app.main:app --reload

The API will be available at:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs
API Endpoint
POST /chat

Example request:

{
  "question": "What is an AI agent?"
}

The response contains:

Generated answer
Retrieved context chunks
Retrieval confidence score
Source document
Page information
Retrieval score
Sample Queries
1. What is an AI agent?

Tests basic knowledge retrieval from the ebook.

2. What are the key characteristics of agentic AI systems?

Tests retrieval of conceptual information.

3. How do AI agents differ from traditional LLM applications?

Tests comparison-based retrieval.

4. What role does planning play in an agentic AI system?

Tests retrieval of a specific concept.

5. Who is the current Prime Minister of India?

This is an out-of-domain question.

Expected behavior:

I could not find this information in the provided knowledge base.

The chatbot should not use external knowledge to answer questions outside the provided ebook.

Grounding Strategy

The chatbot is designed to minimize hallucination by:

Retrieving the most relevant document chunks from Pinecone.
Checking the retrieval relevance score.
Passing only the retrieved context to the LLM.
Instructing the LLM to answer only from the supplied context.
Returning a fallback response when relevant information is not found.
Confidence Score

The confidence value represents the retrieval relevance signal based on the similarity score of the retrieved chunks.

It should not be interpreted as a calibrated probability that the generated answer is correct.

Sample Output

The API returns a response in the following structure:

{
  "answer": "....",
  "context": [
    {
      "content": "....",
      "page": 5
    }
  ],
  "confidence": 0.XX,
  "sources": [
    {
      "document": "Agentic AI Ebook",
      "page": 5,
      "chunk_id": 12,
      "retrieval_score": 0.XX
    }
  ]
}
Evaluation

The following queries were tested through the FastAPI Swagger interface:

Query	Expected Behavior
What is an AI agent?	Answer from ebook
What are the key characteristics of agentic AI systems?	Answer from ebook
How do AI agents differ from traditional LLM applications?	Answer from ebook
What role does planning play in an agentic AI system?	Answer from ebook
Who is the current Prime Minister of India?	Grounded fallback
Limitations
Answers are limited to the information available in the provided ebook.
Retrieval quality depends on chunking and embedding similarity.
The confidence score is a retrieval signal rather than a calibrated probability.
Pinecone and Groq API credentials are required to run the complete pipeline.
Future Improvements
Add a web-based chat UI
Add conversation memory
Improve retrieval using hybrid search or reranking
Add automated evaluation using retrieval and answer quality metrics
Add deterministic document IDs during ingestion
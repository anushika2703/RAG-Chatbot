
# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python, LangGraph, Pinecone, Hugging Face embeddings, Groq, and FastAPI.

This chatbot is designed to answer questions strictly using a provided knowledge base (the Agentic AI ebook). It employs a strict grounding strategy: if the requested information cannot be found in the retrieved context, the system returns a safe fallback response rather than hallucinating from external LLM knowledge.

---

## ✨ Features

* **Document Processing:** Automated PDF ingestion, text extraction, and recursive text chunking with overlap.
* **Semantic Search:** Generates embeddings using Hugging Face (`all-MiniLM-L6-v2`) and performs vector similarity search via Pinecone.
* **Advanced Orchestration:** Utilizes LangGraph for a robust retrieval and generation workflow.
* **High-Speed Generation:** Powered by Groq LLM for fast, accurate answer generation.
* **Strict Grounding:** Employs rigorous prompting to ensure answers rely *only* on the retrieved context. Includes an out-of-domain question fallback.
* **Detailed Context Tracing:** Every response returns the retrieved context, source document name, page information, and a relevance-based confidence score.
* **RESTful API:** Fully interactive backend built on FastAPI.

---

## 🏗️ Architecture

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
                    Pinecone Vector DB
                       |
                       |
User Question --> FastAPI Endpoint
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
             Groq LLM    Fallback Response
                 |
                 v
        Answer + Context + Score
                 |
                 v
               API Response

```

---

## 🛠️ Tech Stack

* **Core Languages & Frameworks:** Python, FastAPI, Pydantic, Uvicorn
* **AI Orchestration:** LangGraph, LangChain
* **Vector Database:** Pinecone
* **Embeddings & LLM:** Hugging Face Sentence Transformers, Groq API

---

## 📂 Project Structure

```text
app/
├── ingestion/
│   └── ingest.py       # PDF loading, chunking, and Pinecone indexing
├── rag/
│   ├── embeddings.py   # HuggingFace embedding configuration
│   ├── graph.py        # LangGraph workflow definition
│   ├── prompts.py      # Grounded LLM prompts
│   └── retriever.py    # Pinecone retrieval logic
├── config.py           # Environment and app configuration
├── main.py             # FastAPI application entry point
└── schemas.py          # Pydantic models for API requests/responses

data/
└── agentic_ai.pdf      # Source knowledge base

```

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG-Chatbot

```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate

```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure environment variables

Create a `.env` file in the root directory of the project. **Do not commit this file to version control.**

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag
GROQ_API_KEY=your_groq_api_key

```

---

## 📖 Usage

### Data Ingestion

Before querying the chatbot, you must ingest the PDF data into your vector database. Place your target ebook at `data/agentic_ai.pdf` and run the ingestion script:

```bash
python -m app.ingestion.ingest

```

*This script handles PDF loading, text extraction, chunking, embedding generation, and Pinecone indexing.*

### Running the API Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --reload

```

* **Base URL:** `[http://127.0.0.1:8000](http://127.0.0.1:8000)`
* **Interactive API Docs (Swagger UI):** `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`

---

## 🔌 API Reference

### `POST /chat`

**Request Payload:**

```json
{
  "question": "What is an AI agent?"
}

```

**Response Payload:**

```json
{
  "answer": "An AI agent is a system that can perceive its environment, make decisions, and take actions to achieve specific goals...",
  "context": [
    {
      "content": "Agents are autonomous entities...",
      "page": 5
    }
  ],
  "confidence": 0.89,
  "sources": [
    {
      "document": "Agentic AI Ebook",
      "page": 5,
      "chunk_id": 12,
      "retrieval_score": 0.89
    }
  ]
}

```

*Note: The **confidence score** represents the retrieval relevance signal based on the vector similarity score of the retrieved chunks. It is not a calibrated probability of the generated answer's factual correctness.*

---

## 🧪 Evaluation & Testing

The chatbot is designed to minimize hallucination. It achieves this by retrieving the most relevant document chunks, verifying the retrieval score, passing strictly that context to the LLM, and forcing a fallback if context is missing.

You can test these behaviors using the `/chat` endpoint:

| Test Query | Goal | Expected Behavior |
| --- | --- | --- |
| *What is an AI agent?* | Basic knowledge retrieval | Answer generated from the ebook. |
| *What are the key characteristics of agentic AI systems?* | Conceptual retrieval | Answer generated from the ebook. |
| *How do AI agents differ from traditional LLM applications?* | Comparison-based retrieval | Answer generated from the ebook. |
| *What role does planning play in an agentic AI system?* | Specific concept retrieval | Answer generated from the ebook. |
| *Who is the current Prime Minister of India?* | Out-of-domain handling | **Fallback:** *"I could not find this information in the provided knowledge base."* |

---

## ⚠️ Limitations

* **Scoped Knowledge:** Answers are strictly limited to the information available in the provided PDF.
* **Retrieval Dependency:** Answer quality depends heavily on chunking strategy and embedding similarity.
* **Score Interpretation:** The confidence score indicates retrieval strength, not absolute factual accuracy.
* **Dependencies:** Valid Pinecone and Groq API credentials are required for the pipeline to function.

---

## 🗺️ Future Improvements

* [ ] Add a web-based chat UI (e.g., Streamlit or Gradio).
* [ ] Implement conversation memory to support follow-up questions.
* [ ] Improve retrieval accuracy using hybrid search (keyword + semantic) or a cross-encoder reranker.
* [ ] Integrate automated evaluation using RAG metrics (e.g., RAGAS) for retrieval and answer quality.
* [ ] Add deterministic document IDs during ingestion to prevent duplicate indexing on rerun.

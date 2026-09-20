from fastapi import FastAPI

from app.rag.graph import graph
from app.schemas import QueryRequest

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG chatbot grounded in the Agentic AI ebook",
    version="1.0.0"
)


@app.get("/")
def root():

    return {
        "message": "Agentic AI RAG API is running"
    }


@app.post("/chat")
def chat(request: QueryRequest):

    result = graph.invoke({

        "question": request.question,

        "documents": [],

        "answer": "",

        "confidence": 0.0,

        "sources": []
    })

    return {
        "answer": result["answer"],

        "context": [
            {
                "content": doc.page_content,
                "page": (
                    doc.metadata.get("page", 0) + 1
                    if doc.metadata.get("page") is not None
                    else None
                )
            }
            for doc in result["documents"]
        ],

        "confidence": result["confidence"],

        "sources": result["sources"]
    }

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Agentic AI RAG Chatbot"
    }
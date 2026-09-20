import os
from typing import TypedDict, List

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END

from app.rag.embeddings import get_embeddings
from app.rag.prompts import SYSTEM_PROMPT

load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

MIN_SCORE = 0.45
TOP_K = 4


# --------------------------------------------------
# RAG State
# --------------------------------------------------

class RAGState(TypedDict):

    question: str
    documents: List[Document]
    answer: str
    confidence: float
    sources: List[dict]


# --------------------------------------------------
# Vector Store
# --------------------------------------------------

embeddings = get_embeddings()

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# Retrieve
# --------------------------------------------------

def retrieve(state: RAGState):

    question = state["question"]

    results = vectorstore.similarity_search_with_score(
        question,
        k=TOP_K
    )

    documents = [doc for doc, score in results]
    scores = [score for doc, score in results]

    confidence = max(scores) if scores else 0.0

    # Build source information
    sources = []

    for doc, score in results:

        page = doc.metadata.get("page")

        # PyPDFLoader uses zero-based page numbers.
        # Convert to human-readable page number.
        if page is not None:
            page = page + 1

        sources.append({
            "source": doc.metadata.get(
                "source",
                "Agentic AI Ebook"
            ),
            "page": page,
            "score": round(float(score), 4)
        })

    return {
        "documents": documents,
        "confidence": confidence,
        "sources": sources
    }


# --------------------------------------------------
# Generate
# --------------------------------------------------

def generate(state: RAGState):

    documents = state["documents"]
    confidence = state["confidence"]
    question = state["question"]

    # --------------------------------------------------
    # Reject if nothing relevant was retrieved
    # --------------------------------------------------

    if not documents or confidence < MIN_SCORE:

        return {
            "answer": (
                "I could not find this information "
                "in the provided knowledge base."
            )
        }

    # --------------------------------------------------
    # Build context
    # --------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = SYSTEM_PROMPT.format(
        context=context
    )

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    response = llm.invoke([
        ("system", prompt),
        ("human", question)
    ])

    return {
        "answer": response.content
    }


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

graph_builder = StateGraph(RAGState)

graph_builder.add_node(
    "retrieve",
    retrieve
)

graph_builder.add_node(
    "generate",
    generate
)

graph_builder.add_edge(
    START,
    "retrieve"
)

graph_builder.add_edge(
    "retrieve",
    "generate"
)

graph_builder.add_edge(
    "generate",
    END
)

graph = graph_builder.compile()


# --------------------------------------------------
# Local Test
# --------------------------------------------------

if __name__ == "__main__":

    result = graph.invoke({

        "question": "What is an AI agent?",

        "documents": [],

        "answer": "",

        "confidence": 0.0,

        "sources": []
    })

    print("\nANSWER:")
    print(result["answer"])

    print("\nCONFIDENCE:")
    print(result["confidence"])

    print("\nSOURCES:")

    for source in result["sources"]:
        print(source)
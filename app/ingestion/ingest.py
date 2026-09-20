import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.rag.embeddings import get_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

PDF_PATH = "data/agentic_ai.pdf"


def ingest():

    # -------------------------
    # 1. Load PDF
    # -------------------------

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    # -------------------------
    # 2. Split into chunks
    # -------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # -------------------------
    # 3. Add metadata
    # -------------------------

    for i, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = "Agentic AI Ebook"

    # -------------------------
    # 4. Pinecone
    # -------------------------

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index["name"] for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:

        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    # -------------------------
    # 5. Create vector store
    # -------------------------

    embeddings = get_embeddings()

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print("Successfully indexed PDF")


if __name__ == "__main__":
    ingest()
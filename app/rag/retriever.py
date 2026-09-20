import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

from app.rag.embeddings import get_embeddings

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


def get_retriever():

    embeddings = get_embeddings()

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )
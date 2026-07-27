"""
LangGraph-compatible tools for the meeting-notes chatbot.

Each function decorated with @tool can be called by the LLM agent. The module
initialises the Chroma vector-store and HuggingFace embeddings once at import
time so they are available to every tool invocation without Streamlit context.
"""

import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings

"""
    Create a Chroma vector store and HuggingFace embeddings instance at the module level.
    This avoids using @st.cache_resource, which requires a Streamlit session context that isn't available when LangGraph's tool node invokes these tools.
"""

# Shared embedding model — same one used during document ingestion (rag_document_loader.py)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Persistent Chroma instance pointing at the local vector database
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

@tool
def query_documents(question):
    """
    Query the Chroma vector store for relevant documents.

    Args:
        question (str): The user's question might be answerable from the documents.

    Returns:
        list: A list of relevant document contents that match with the question using RAG.
    """
    # Retrieve the top-5 most similar document chunks from Chroma
    # similarity_docs = db.similarity_search(question, k=5)
    similarity_docs = db.max_marginal_relevance_search(question, k=3, fetch_k=10)

    # Format each result with its source filename and content for the LLM to read
    docs_formated = [
        f"Source: {doc.metadata.get('source', 'NA')}\n"
        f"Content: {doc.page_content}"
        for doc in similarity_docs
    ]

    return "\n\n---\n\n".join(docs_formated)

@tool
def get_current_time() ->str:
    """Returns the current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Registry of tools exposed to the LangGraph agent (name → callable)
available_functions = {
    "get_current_time": get_current_time,
    "query_documents": query_documents
}

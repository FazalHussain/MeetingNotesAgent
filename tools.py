import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def get_chroma_instance():
    """
    Create and cache the Chroma instance.

    Streamlit caches this object so the Chroma instance is created only once,
    even when the app reruns after each user interaction.
    """

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


db = get_chroma_instance()

@tool
def query_documents(question):
    """
    Query the Chroma vector store for relevant documents.

    Args:
        question (str): The user's question might be answerable from the documents.

    Returns:
        list: A list of relevant document contents that match with the question using RAG.
    """
    similarity_docs = db.similarity_search(question, k=3)
    docs_formated = list(map(lambda docs: f"Source: {docs.metadata.get('source', 'NA')}\nContent: {docs.page_content}", similarity_docs))
    return str(docs_formated)

@tool
def get_current_time() ->str:
    """Returns the current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

available_functions = {
    "get_current_time": get_current_time,
    "query_documents": query_documents
}
"""
Document loader and vector store builder.

This script reads meeting-note .txt files from a configurable directory,
splits them into chunks, computes HuggingFace embeddings, and persists the
resulting Chroma vector store to ./chroma_db for later retrieval by the RAG
pipeline (tools.py).
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Directory containing meeting note .txt files (default: "meeting_notes")
rag_directory = os.getenv("RAG_DIRECTORY", "meeting_notes")

def load_documents_from_directory(directory):
    """
    Load documents from the specified directory.

    Args:
        directory_path: The path to the directory containing documents.
    """
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    documents = loader.load()

    # Split the documents into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,   
        chunk_overlap=80, 
        separators=["\n\n", "\n", " "],
    )
    documents = text_splitter.split_documents(documents)
    return documents

def main():
    """Load, split, embed, and persist meeting notes to a Chroma vector store."""
    # Load and chunk documents from the configured directory
    documents = load_documents_from_directory(rag_directory)

    # Compute sentence-level embeddings using a lightweight transformer model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store the embedded chunks in a local Chroma database on disk
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

if __name__ == "__main__":
    main()
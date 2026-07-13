from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

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
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    return documents

def main():
    # Get the documents from the specified directory
    documents = load_documents_from_directory(rag_directory)

    # Create embeddings for the documents using SentenceTransformer
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the documents into a Chroma vector store and save it to disk
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

if __name__ == "__main__":
    main()
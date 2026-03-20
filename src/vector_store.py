import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def get_embedding_function():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

def get_vector_store():
    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embedding_function()
    )

def ingest_schema(text_content: str):
    """
    Ingests the schema text into the vector store.
    WARNING: This clears the existing vector store to ensure no duplicates.
    """
    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    # Convert to absolute path to avoid ambiguity
    persist_directory = os.path.abspath(persist_directory)

    # Clear existing data for a fresh ingestion by removing the entire directory
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            print(f"Cleared existing vector store at {persist_directory}")
        except Exception as e:
            print(f'Warning: Failed to delete {persist_directory}. Reason: {e}')
            # If we can't delete the whole directory, try deleting its contents
            for filename in os.listdir(persist_directory):
                file_path = os.path.join(persist_directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as ex:
                    print(f'Failed to delete {file_path}. Reason: {ex}')
    
    # Ensure the directory exists
    os.makedirs(persist_directory, exist_ok=True)

    vector_store = get_vector_store()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    # Check if text_content is empty
    if not text_content.strip():
        print("Warning: No schema content to ingest.")
        return

    splits = text_splitter.split_text(text_content)

    # Chroma handles persistence automatically in newer versions,
    # but we add texts here.
    vector_store.add_texts(splits)
    print(f"Ingested {len(splits)} chunks into vector store at {persist_directory}")

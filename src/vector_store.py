import os
import requests
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
FALLBACK_EMBEDDING_MODELS = [
    "models/gemini-embedding-001",
    "models/gemini-embedding-2-preview",
]
COLLECTION_NAME = "langchain"

# Single shared client instance to avoid SQLite locking issues
_chroma_client = None


def _get_persist_directory():
    """Returns the absolute path to the ChromaDB persist directory."""
    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    return os.path.abspath(persist_directory)


def _get_chroma_client():
    """Return a singleton PersistentClient so every caller shares the same
    SQLite connection — avoids 'readonly database' locking errors."""
    global _chroma_client
    if _chroma_client is None:
        persist_directory = _get_persist_directory()
        os.makedirs(persist_directory, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=persist_directory)
    return _chroma_client


def fetch_embedding_models() -> list[str]:
    """Fetch available embedding models live from Google's API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return FALLBACK_EMBEDDING_MODELS
    try:
        resp = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10,
        )
        resp.raise_for_status()
        models = []
        for m in resp.json().get("models", []):
            if "embedContent" in m.get("supportedGenerationMethods", []):
                models.append(m["name"])
        return sorted(models) if models else FALLBACK_EMBEDDING_MODELS
    except Exception:
        return FALLBACK_EMBEDDING_MODELS


def get_embedding_function(model: str = None):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    model = model or DEFAULT_EMBEDDING_MODEL
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def get_vector_store(embedding_model: str = None):
    client = _get_chroma_client()
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(embedding_model),
    )


def ingest_schema(text_content: str, embedding_model: str = None):
    """
    Ingests the schema text into the vector store.
    Clears the existing collection first to ensure no duplicates.
    """
    if not text_content.strip():
        print("Warning: No schema content to ingest.")
        return

    client = _get_chroma_client()

    # Delete existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Cleared existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass  # Collection doesn't exist yet

    # Create fresh vector store on the same client
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(embedding_model),
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    splits = text_splitter.split_text(text_content)
    vector_store.add_texts(splits)
    print(f"Ingested {len(splits)} chunks into vector store at {_get_persist_directory()}")

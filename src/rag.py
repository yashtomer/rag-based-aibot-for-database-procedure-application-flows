from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.vector_store import get_vector_store
import os
import requests
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDERS = {
    "Gemini (Google)": {
        "env_key": "GOOGLE_API_KEY",
    },
    "Groq": {
        "env_key": "GROQ_API_KEY",
    },
}

FALLBACK_MODELS = {
    "Gemini (Google)": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
    "Groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
}


def fetch_models(provider: str) -> tuple[list[str], dict]:
    """Fetch available models live from the provider API.
    Returns (sorted model names, {model_name: details_dict})."""
    api_key = os.getenv(LLM_PROVIDERS[provider]["env_key"])
    if not api_key:
        return FALLBACK_MODELS.get(provider, []), {}

    try:
        if provider == "Gemini (Google)":
            resp = requests.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                timeout=10,
            )
            resp.raise_for_status()
            models = []
            details = {}
            for m in resp.json().get("models", []):
                name = m["name"].removeprefix("models/")
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    models.append(name)
                    details[name] = {
                        "display_name": m.get("displayName", name),
                        "description": m.get("description", "N/A"),
                        "context_window": m.get("inputTokenLimit", "N/A"),
                        "max_output_tokens": m.get("outputTokenLimit", "N/A"),
                        "owner": "Google",
                    }
            return (sorted(models), details) if models else (FALLBACK_MODELS[provider], {})

        elif provider == "Groq":
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            models = []
            details = {}
            for m in resp.json().get("data", []):
                if m.get("active", True):
                    mid = m["id"]
                    models.append(mid)
                    details[mid] = {
                        "display_name": mid,
                        "description": f"Owned by {m.get('owned_by', 'N/A')}",
                        "context_window": m.get("context_window", "N/A"),
                        "max_output_tokens": m.get("max_completion_tokens", "N/A"),
                        "owner": m.get("owned_by", "N/A"),
                    }
            return (sorted(models), details) if models else (FALLBACK_MODELS[provider], {})

    except Exception:
        return FALLBACK_MODELS.get(provider, []), {}


def get_llm(provider: str = "Gemini (Google)", model: str = None):
    config = LLM_PROVIDERS[provider]
    api_key = os.getenv(config["env_key"])
    if not api_key:
        raise ValueError(f"{config['env_key']} environment variable is not set")

    model = model or FALLBACK_MODELS[provider][0]

    if provider == "Gemini (Google)":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0,
            timeout=30,
            max_retries=2,
        )
    else:
        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=0,
            max_retries=2,
        )

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def _retriever_k(provider: str) -> int:
    """Return fewer chunks for providers with smaller context windows."""
    return 3 if provider == "Groq" else 5


def get_chat_response(query: str, provider: str = "Gemini (Google)", model: str = None, embedding_model: str = None) -> str:
    """
    Retrieves context and answers a natural language question about the database.
    """
    vector_store = get_vector_store(embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": _retriever_k(provider)})
    llm = get_llm(provider, model)

    template = """You are an expert database engineer and architect.
    Use the following pieces of retrieved database schema context to answer the user's question.
    If the context doesn't contain the answer, say you don't have enough information.
    Be precise and refer to specific table names and columns where appropriate.

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)

def get_diagram_response(query: str, provider: str = "Gemini (Google)", model: str = None, embedding_model: str = None) -> str:
    """
    Generates Mermaid.js diagram syntax based on the request and schema context.
    """
    vector_store = get_vector_store(embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": _retriever_k(provider)})
    llm = get_llm(provider, model)

    template = """You are an expert in data visualization and Mermaid.js.
    Use the following database schema context to generate a Mermaid diagram that satisfies the user's request.

    The output must be ONLY the Mermaid code block, starting with ```mermaid and ending with ```.
    Do not include any other text or explanation.

    If the request implies an Entity Relationship Diagram (ERD), use `erDiagram`.
    If the request implies a flow or process, use `graph TD` or `sequenceDiagram` as appropriate.

    Context:
    {context}

    Request: {question}

    Mermaid Diagram:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(query)

    # Clean up response to ensure it's just the code if the LLM is chatty
    # But prompt says ONLY, usually reliable.
    # We can strip the markdown blocks if needed by the frontend, but returning them is fine too.
    return response

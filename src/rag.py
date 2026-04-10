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
    "OpenAI": {
        "env_key": "OPENAI_API_KEY",
    },
    "Anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
    },
}

FALLBACK_MODELS = {
    "Gemini (Google)": ["gemini-3.1-pro", "gemini-3.1-flash", "gemini-2.5-pro", "gemini-2.5-flash","gemini-2.5-flash-lite"],
    "Groq": ["llama-4-70b", "llama-4-8b", "deepseek-r1-distill-llama-70b", "mistral-saba-24b"],
    "OpenAI": ["gpt-5.4-pro", "gpt-5.4-thinking", "gpt-5.4-mini"],
    "Anthropic": ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001", "claude-sonnet-3-7"],
}


def fetch_models(provider: str, api_key: str = None) -> tuple[list[str], dict]:
    """Fetch available models live from the provider API.
    Returns (sorted model names, {model_name: details_dict})."""
    api_key = api_key or os.getenv(LLM_PROVIDERS[provider]["env_key"])
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

        elif provider == "OpenAI":
            resp = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            models = []
            details = {}
            # Filter for common chat models to avoid clutter
            chat_patterns = ["gpt-5", "gpt-4", "o1-"]
            for m in resp.json().get("data", []):
                mid = m["id"]
                if any(p in mid for p in chat_patterns):
                    models.append(mid)
                    details[mid] = {
                        "display_name": mid,
                        "description": f"OpenAI Chat Model",
                        "context_window": "N/A",  # OpenAI API doesn't provide this in models list
                        "max_output_tokens": "N/A",
                        "owner": "OpenAI",
                    }
            return (sorted(models), details) if models else (FALLBACK_MODELS[provider], {})

        elif provider == "Anthropic":
            # Anthropic doesn't have a simple public 'list models' endpoint like OpenAI/Groq
            # that returns all available chat models with detailed info.
            # We will use the fallback list but we could attempt to use their Models API if available.
            return (sorted(FALLBACK_MODELS[provider]), {})

        elif provider == "Groq":
            return (sorted(FALLBACK_MODELS[provider]), {})

    except Exception as e:
        print(f"Error fetching models for {provider}: {e}")
        return FALLBACK_MODELS.get(provider, []), {}

    # Safety fallback for any unhandled provider branch.
    return FALLBACK_MODELS.get(provider, []), {}


def get_llm(provider: str = "Gemini (Google)", model: str = None, api_key: str = None):
    config = LLM_PROVIDERS[provider]
    api_key = api_key or os.getenv(config["env_key"])
    if not api_key:
        raise ValueError(f"API key for {provider} is not set. Please provide it in the UI or set {config['env_key']} in .env")

    model = model or FALLBACK_MODELS[provider][0]

    if provider == "Gemini (Google)":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0,
            timeout=30,
            max_retries=2,
        )
    elif provider == "Groq":
        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=0,
            max_retries=2,
        )
    elif provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0,
            max_retries=2,
        )
    elif provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
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


def get_chat_response(query: str, provider: str = "Gemini (Google)", model: str = None, embedding_model: str = None, api_key: str = None) -> str:
    """
    Retrieves context and answers a natural language question about the database.
    """
    vector_store = get_vector_store(embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": _retriever_k(provider)})
    llm = get_llm(provider, model, api_key)

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

def get_diagram_response(query: str, provider: str = "Gemini (Google)", model: str = None, embedding_model: str = None, api_key: str = None) -> str:
    """
    Generates Mermaid.js diagram syntax based on the request and schema context.
    Uses full database schema for complete ER diagrams.
    """
    from src.database import get_full_db_context

    # For ER diagrams, use the full schema so no tables are missed
    full_db_keywords = ["entire", "full", "complete", "all tables", "whole", "database"]
    _use_full_schema = any(kw in query.lower() for kw in full_db_keywords) or "er diagram" in query.lower() or "erd" in query.lower()

    if _use_full_schema:
        try:
            context = get_full_db_context()
        except Exception:
            context = None

    if not _use_full_schema or not context:
        vector_store = get_vector_store(embedding_model)
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(query)
        context = format_docs(docs)

    llm = get_llm(provider, model, api_key)

    template = """You are an expert in data visualization and Mermaid.js.
    Use the following database schema context to generate a Mermaid diagram that satisfies the user's request.
    IMPORTANT: Include ALL tables and ALL relationships from the schema. Do not skip any table.

    The output must be ONLY the Mermaid code block, starting with ```mermaid and ending with ```.
    Do not include any other text or explanation.

    If the request implies an Entity Relationship Diagram (ERD), use `erDiagram`.
    If the request implies a flow or process, use `graph TD` or `sequenceDiagram` as appropriate.

    For erDiagram, use this format for each table:
    TableName {{
        type column_name PK "comment"
        type column_name FK
        type column_name
    }}

    And show relationships like:
    TableA ||--o{{ TableB : "foreign_key"

    Context:
    {context}

    Request: {question}

    Mermaid Diagram:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": query})

    return response

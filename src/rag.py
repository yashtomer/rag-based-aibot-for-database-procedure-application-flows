from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.vector_store import get_vector_store
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    return ChatGoogleGenerativeAI(model="models/gemini-flash-latest", google_api_key=api_key, temperature=0)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def get_chat_response(query: str) -> str:
    """
    Retrieves context and answers a natural language question about the database.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

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

def get_diagram_response(query: str) -> str:
    """
    Generates Mermaid.js diagram syntax based on the request and schema context.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = get_llm()

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

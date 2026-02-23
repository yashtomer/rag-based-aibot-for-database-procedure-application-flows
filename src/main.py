from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.database import get_full_db_context
from src.vector_store import ingest_schema
from src.rag import get_chat_response, get_diagram_response
import re

app = FastAPI(title="DB Intelligence Bot")

class IngestResponse(BaseModel):
    status: str
    message: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

class DiagramRequest(BaseModel):
    request: str

class DiagramResponse(BaseModel):
    mermaid_code: str

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint():
    try:
        print("Starting ingestion...")
        context = get_full_db_context()
        if not context or "Error reflecting database" in context:
            # Check if it's just empty or actual error
            if "Error reflecting database" in context:
                raise HTTPException(status_code=500, detail=f"Database introspection failed: {context}")

        ingest_schema(context)
        return IngestResponse(status="success", message="Database schema ingested successfully")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        response = get_chat_response(req.query)
        return ChatResponse(answer=response)
    except Exception as e:
        print(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diagram", response_model=DiagramResponse)
async def diagram_endpoint(req: DiagramRequest):
    try:
        response = get_diagram_response(req.request)

        # Clean up markdown code blocks if present
        clean_response = response
        if "```mermaid" in response:
            clean_response = response.split("```mermaid")[1].split("```")[0].strip()
        elif "```" in response:
            clean_response = response.split("```")[1].split("```")[0].strip()

        return DiagramResponse(mermaid_code=clean_response)
    except Exception as e:
        print(f"Diagram generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

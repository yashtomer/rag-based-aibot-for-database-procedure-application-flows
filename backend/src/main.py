from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.database import get_full_db_context
from src.vector_store import ingest_schema
from src.rag import get_chat_response, get_diagram_response
from src.seeder import seed_database
import re

# Auto-run database verification and administrator credentials seeding on boot
seed_database()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DB Intelligence Bot")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local docker compose setup
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    database: str = None

class IngestResponse(BaseModel):
    status: str
    message: str

@app.get("/db/status")
async def db_status_endpoint(database: str = None):
    import os
    db_name = database if database else os.getenv("MYSQL_DATABASE", "").strip('\"\'')
    host = os.getenv("MYSQL_HOST", "").strip('\"\'')
    
    try:
        from src.database import get_engine
        from sqlalchemy import inspect, text
        
        # 1. Connect to specific database to verify active connection
        engine = get_engine(db_name)
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        status = "Connected"
        
        # 2. Query other databases available on the MySQL server
        databases = [db_name] if db_name else []
        try:
            with engine.connect() as conn:
                res = conn.execute(text("SHOW DATABASES"))
                databases = [row[0] for row in res if row[0] not in ('information_schema', 'performance_schema', 'mysql', 'sys')]
        except Exception:
            pass
            
    except Exception as e:
        table_names = []
        status = f"Offline (Error: {str(e)})"
        databases = [db_name] if db_name else ["sqlite_memory"]
            
    return {
        "active_db": db_name if db_name else "sqlite_memory",
        "host": host if host else "in-memory",
        "status": status,
        "tables": table_names,
        "databases": databases if databases else [db_name]
    }

@app.get("/db/table/{table_name}")
async def get_table_details_endpoint(table_name: str, database: str = None):
    try:
        from src.database import get_engine
        from sqlalchemy import inspect, text
        
        engine = get_engine(database)
        inspector = inspect(engine)
        
        columns = inspector.get_columns(table_name)
        fks = inspector.get_foreign_keys(table_name)
        
        formatted_columns = []
        for col in columns:
            fk_info = ""
            for fk in fks:
                if col['name'] in fk['constrained_columns']:
                    ref_cols = ", ".join(fk['referred_columns'])
                    fk_info = f"FK -> {fk['referred_table']}({ref_cols})"
            
            formatted_columns.append({
                "name": col['name'],
                "type": str(col['type']),
                "primary": col.get('primary_key', False),
                "nullable": col.get('nullable', True),
                "extra": fk_info
            })
            
        # Live row count
        row_count = 0
        try:
            with engine.connect() as conn:
                count_res = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`"))
                row_count = count_res.scalar()
        except Exception:
            row_count = 0
            
        return {
            "name": table_name,
            "rows": row_count,
            "columns": formatted_columns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/auth/login")
async def login_endpoint(req: LoginRequest):
    from src.seeder import authenticate_user
    user_info = authenticate_user(req.email, req.password)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return user_info

class TestConnectionRequest(BaseModel):
    provider: str
    model: str
    api_key: str = None

class ChatRequest(BaseModel):
    query: str
    provider: str = None
    model: str = None
    api_key: str = None

class ChatResponse(BaseModel):
    answer: str

class DiagramRequest(BaseModel):
    request: str
    provider: str = None
    model: str = None
    api_key: str = None

class DiagramResponse(BaseModel):
    mermaid_code: str

@app.post("/test-connection")
async def test_connection_endpoint(req: TestConnectionRequest):
    try:
        from src.rag import get_llm
        provider_norm = "Gemini (Google)"
        if req.provider and "groq" in req.provider.lower():
            provider_norm = "Groq"
        elif req.provider and "openai" in req.provider.lower():
            provider_norm = "OpenAI"
        elif req.provider and "anthropic" in req.provider.lower():
            provider_norm = "Anthropic"
            
        llm = get_llm(provider_norm, req.model, req.api_key)
        try:
            llm.invoke("Say 'OK'")
            return {"success": True, "message": f"Successfully connected to {req.provider} using model {req.model}!"}
        except Exception as e:
            return {"success": False, "message": f"Handshake failed: {str(e)}"}
    except Exception as e:
        return {"success": False, "message": f"Configuration failed: {str(e)}"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest = None):
    try:
        print("Starting ingestion...")
        db_override = req.database if req else None
        context = get_full_db_context(db_override)
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
        provider_norm = "Gemini (Google)"
        if req.provider and "groq" in req.provider.lower():
            provider_norm = "Groq"
        elif req.provider and "openai" in req.provider.lower():
            provider_norm = "OpenAI"
        elif req.provider and "anthropic" in req.provider.lower():
            provider_norm = "Anthropic"

        response = get_chat_response(
            query=req.query,
            provider=provider_norm,
            model=req.model,
            api_key=req.api_key
        )
        return ChatResponse(answer=response)
    except Exception as e:
        print(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diagram", response_model=DiagramResponse)
async def diagram_endpoint(req: DiagramRequest):
    try:
        provider_norm = "Gemini (Google)"
        if req.provider and "groq" in req.provider.lower():
            provider_norm = "Groq"
        elif req.provider and "openai" in req.provider.lower():
            provider_norm = "OpenAI"
        elif req.provider and "anthropic" in req.provider.lower():
            provider_norm = "Anthropic"

        response = get_diagram_response(
            query=req.request,
            provider=provider_norm,
            model=req.model,
            api_key=req.api_key
        )

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

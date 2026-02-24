# Database Intelligence Bot

A RAG-based AI bot designed to assist with database-related tasks using Google Gemini and ChromaDB. It can answer natural language questions about your database schema and generate Mermaid.js diagrams.

## Prerequisites

- **Python 3.10+**
- **MySQL Database**
- **Google Gemini API Key**
- **uv** (An extremely fast Python package installer and resolver)

## Installation

1. Clone the repository.
2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```
   This will create a `.venv` directory and install all required packages.

## Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` with your database credentials and Google API Key.
   - `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`
   - `GOOGLE_API_KEY`

## Running the Application

Start the FastAPI server using the provided script:
```bash
./run.sh
```

Or manually with `uv`:
```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`.

## API Usage

### 1. Ingest Database Schema

Run this first to scan your database and build the knowledge base.

```bash
curl -X POST http://localhost:8000/ingest
```

### 2. Chat with the Bot

Ask questions about table relationships, stored procedures, or general queries.

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Explain how the orders table relates to customers"}'
```

### 3. Generate Diagrams

Request Mermaid.js syntax for ER diagrams or flows.

```bash
curl -X POST http://localhost:8000/diagram \
     -H "Content-Type: application/json" \
     -d '{"request": "Create an ER diagram for the user management module"}'
```

## Project Structure

- `src/database.py`: Handles database connection and schema introspection using SQLAlchemy.
- `src/vector_store.py`: Manages vector embeddings and retrieval using ChromaDB.
- `src/rag.py`: Contains the logic for interacting with the LLM (Gemini) using LangChain.
- `src/main.py`: The FastAPI application defining the endpoints.

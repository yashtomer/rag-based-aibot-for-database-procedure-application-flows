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

### Option 1: Development (Local)

1. **Start the Streamlit UI (Frontend)**
   ```bash
   uv run streamlit run app.py
   ```
   The UI will be available at `http://localhost:8501`.

2. **Start the FastAPI Backend (Optional)**
   ```bash
   ./run.sh
   # or manually: uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   The API will be available at `http://localhost:8000`.

### Option 2: Setup via Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t db-intelligence-bot .
   ```
2. **Run the Docker container:**
   ```bash
   docker run -d \
     --name db-bot-container \
     -p 8502:8501 \
     --env-file .env \
     -v $(pwd)/chroma_db:/app/chroma_db \
     db-intelligence-bot
   ```
   The application UI will now be accessible at `http://localhost:8501`.

### Option 3: Setup via Linux (Production)

To deploy securely for production on a Linux server without containerization:

1. **Install global requirements (Ubuntu/Debian):**
   ```bash
   sudo apt update
   sudo apt install nginx certbot python3-certbot-nginx
   ```
2. **Create a Systemd service for Streamlit (`/etc/systemd/system/dbbot.service`):**
   ```ini
   [Unit]
   Description=Streamlit Database Bot Interface
   After=network.target

   [Service]
   User=your_linux_user
   WorkingDirectory=/path/to/your/project
   EnvironmentFile=/path/to/your/project/.env
   ExecStart=/path/to/your/project/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   Restart=always
   RestartSec=3

   [Install]
   WantedBy=multi-user.target
   ```
3. **Start and enable the Streamlit service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start dbbot
   sudo systemctl enable dbbot
   ```
4. **Configure Nginx as a WebSockets Reverse Proxy (`/etc/nginx/sites-available/dbbot`):**
   ```nginx
   server {
       listen 80;
       server_name your_domain_or_ip.com;

       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header Host $host;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_read_timeout 86400;
       }
   }
   ```
5. **Enable Nginx configuration and apply SSL:**
   ```bash
   sudo ln -s /etc/nginx/sites-available/dbbot /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   
   # Setup free standard SSL Certificates
   sudo certbot --nginx -d your_domain_or_ip.com
   ```
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

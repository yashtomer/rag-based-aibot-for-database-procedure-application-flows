# 🤖 Aeologic Database Intelligence Bot Console

A state-of-the-art, secure, RAG-based intelligence engine powered by LangChain, ChromaDB, and a dynamic multi-model LLM gateway (Gemini, OpenAI, Anthropic, and Groq). This system introspects complex MySQL database schemas, analyses tables and relationships, generates high-fidelity Entity Relationship Diagrams (ERDs) using Mermaid.js, and produces documentation in real time.

All styled in a gorgeous, atmospheric **"Dark Luxe"** glassmorphic dashboard!

---

## ✨ System Features

* **Multi-LLM Provider Suite**: Switch in real time between Google Gemini, OpenAI, Anthropic, and Groq.
* **Custom Models**: Type in any proprietary or custom model name (e.g., `gpt-4.5-turbo`, `claude-3-7-sonnet-latest`) on the fly.
* **Transient Session Security**: API Keys are saved **only in your transient session storage**; closing your tab instantly destroys the keys, ensuring zero persistent credentials.
* **Auto-Created Databases & Tables**: Connects to the server, creates the target database if not found, sets up the `users` table, and automatically seeds the credentials configured in `.env`.
* **Database-Backed Authentication**: High-fidelity dynamic authentication panel featuring a password eye visibility toggle and instant MySQL record validation.

---

## 📁 Workspace Directory Structure

Below is the directory schema of the project, highlighting the core components:

```
rag-based-aibot-for-database-procedure-application-flows/
├── backend/                       # FastAPI Server Root Node
│   ├── src/
│   │   ├── database.py            # MySQL schema introspection & connection engine
│   │   ├── seeder.py              # Auto-creates target DB, users table, and seeds admin
│   │   ├── vector_store.py        # Schema RAG vector indexing using ChromaDB
│   │   ├── rag.py                 # Multi-LLM provider orchestration (Gemini, OpenAI, etc.)
│   │   └── main.py                # API endpoints & auth routing (/auth/login, /chat, etc.)
│   ├── Dockerfile                 # Slim-Python 3.12 Docker execution manifest
│   └── pyproject.toml             # uv package dependencies
│
├── frontend/                      # React Vite Client Root
│   ├── public/
│   │   ├── favicon.ico            # Official brand tab icon
│   │   └── logo-white.svg         # White premium logo asset (Dark Mode)
│   │   └── logo.svg               # Colored premium logo asset (Light Mode)
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.jsx        # Premium Sidebar with theme-aware branding
│   │   │   └── Navbar.jsx         # Header containing live DB & theme toggle
│   │   ├── pages/
│   │   │   ├── LoginPage.jsx      # Auth panel with Password Eye visibility toggle
│   │   │   ├── DashboardPage.jsx  # Chat console, prompt engineering & model selector
│   │   │   └── IngestionPage.jsx  # DB Reflection & vector indexing cockpit
│   │   ├── App.jsx                # Global router and session cache migrator
│   │   └── index.css              # Custom HSL-based design system tokens
│   ├── Dockerfile                 # Frontend build/serve container setup
│   └── package.json               # Node dependencies
│
├── .env                           # Local operational environment configurations
├── .env.example                   # Reference environment variables
└── docker-compose.yml             # Full Multi-Container Orchestration file
```

---

## ⚙️ Environment Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Update the configuration keys inside `.env`:
   ```ini
   # Database Configuration
   MYSQL_HOST=host.docker.internal
   MYSQL_PORT=3306
   MYSQL_USER=root
   MYSQL_PASSWORD=your_mysql_password
   MYSQL_DATABASE=ragbasedsql

   # Vector DB Location (default: ./chroma_db)
   CHROMA_PERSIST_DIRECTORY=./chroma_db

   # Default Admin Seeder Credentials
   ADMIN_EMAIL=admin@aeologic.com
   ADMIN_PASSWORD="your_admin_password"
   ADMIN_NAME="Aeologic User"
   ```

---

## 🚀 Running the Application

### Option 1: Setup via Docker Compose (Recommended - Fully Automated)

Docker Compose automatically spins up the MySQL-server wrapper, boots the FastAPI backend, runs database seeding, and serves the React Vite UI:

1. **Build and start services**:
   ```bash
   docker compose up --build -d
   ```
2. **Review Seeding & Start Logs**:
   ```bash
   docker compose logs -f backend
   ```
3. **Access the Console**:
   * **Console UI**: Open `http://localhost:8502` in your browser.
   * **Backend REST API**: Running at `http://localhost:8000`.

---

### Option 2: Local Development (Without Containers)

#### 1. Start the FastAPI Backend
```bash
cd backend
# Install dependencies using uv into virtual environment
uv sync
# Execute the startup script (runs seeder & boots uvicorn)
./run.sh
```
*The API node will listen at `http://localhost:8000`.*

#### 2. Start the React Client
```bash
cd frontend
# Install package dependencies
npm install
# Start the local hot-reloaded development server
npm run dev
```
*The development client will listen at `http://localhost:8502`.*

---

## 🌐 Option 3: High-Performance Linux Production Deployment

For enterprise deployments on a bare Linux server (Ubuntu/Debian) using **PM2** (Process Manager 2) and **Nginx**:

### 1. Install System Requirements
```bash
sudo apt update
sudo apt install -y curl git nginx nodejs npm python3 python3-pip
# Install PM2 globally
sudo npm install -y pm2 -g
# Install uv globally for fast python packaging
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Configure Backend Service under PM2
Build your virtual environment, execute the migrations, and daemonize the FastAPI application using PM2:
```bash
cd backend
uv sync

# Start FastAPI under PM2
pm2 start "uv run uvicorn src.main:app --host 127.0.0.1 --port 8000" --name "db-bot-backend"
pm2 save
```

### 3. Deploy and Serve React Production Build
Compile the frontend client to highly compressed production assets and daemonize using PM2's static host or Nginx directly:
```bash
cd ../frontend
npm install
# Build distribution bundle
npm run build

# Start simple static server using PM2
pm2 serve dist 8502 --name "db-bot-frontend" --spa
pm2 save
```

### 4. Setup Nginx Reverse Proxy with SSL
1. **Configure WebSockets Proxy** inside `/etc/nginx/sites-available/dbbot`:
   ```nginx
   server {
       listen 80;
       server_name your_domain_or_ip.com;

       # Serve React Frontend UI
       location / {
           proxy_pass http://127.0.0.1:8502;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }

       # Route REST API requests to FastAPI backend
       location /auth/ {
           proxy_pass http://127.0.0.1:8000/auth/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       }

       location /db/ {
           proxy_pass http://127.0.0.1:8000/db/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       }

       location /chat {
           proxy_pass http://127.0.0.1:8000/chat;
           proxy_set_header Host $host;
       }

       location /diagram {
           proxy_pass http://127.0.0.1:8000/diagram;
           proxy_set_header Host $host;
       }
   }
   ```
2. **Enable and Apply Configuration**:
   ```bash
   sudo ln -s /etc/nginx/sites-available/dbbot /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```
3. **Bind Standard Free SSL Certificates**:
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your_domain_or_ip.com
   ```

---

## 🔒 Security Operations & API Endpoints

### 🔑 Authentication Handshake
```bash
curl -X POST http://localhost:8000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email": "admin@aeologic.com", "password": "Admin@#12345"}'
```

### ⚡ Database Schema Ingestion
```bash
curl -X POST http://localhost:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"database": "ragbasedsql"}'
```

### 💬 Natural Language Prompt Chat
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How many tables are present in the target schema?",
       "provider": "Anthropic",
       "model": "claude-3-7-sonnet-latest",
       "api_key": "your_secure_session_api_key"
     }'
```

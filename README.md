We are looking for an experienced Python Backend Engineer to design and implement a RAG (Retrieval-Augmented Generation) based AI bot that deeply understands our databases, procedures, and application flows, and can assist with all types of database-related tasks.
 
The goal is to build an internal intelligence bot that can:
 
Answer natural-language questions about our databases
Explain stored procedures, jobs, and workflows
Generate architecture diagrams, ER diagrams, and flow charts
Help write and validate SQL queries
Assist in documentation, reporting, and impact analysis
This bot will be used by developers, DevOps, and analysts as a single interface to understand and interact with our systems.
 
Core Objectives
 
1. Knowledge Ingestion (RAG Layer)
The bot should ingest and index:
 
Database schemas (tables, columns, relationships)
Stored procedures, functions, triggers, jobs
Existing SQL scripts
Application documentation
Admin panel procedures and workflows
Sample reports and queries
This knowledge should be searchable and retrievable using vector embeddings.
 
2. Database Intelligence
The bot must be able to:
 
Explain table relationships and dependencies
Answer questions like:
“Which tables are impacted if I change column X?”
“How does this stored procedure work step-by-step?”
“Which procedures write to this table?”
Generate optimized SQL queries from natural language
Validate queries against schema rules
Help analyze performance and indexing
 
3. Architecture & Flow Visualization
The bot should generate or assist in generating:
 
ER Diagrams
Database schema diagrams
Procedure execution flows
Data movement flow charts
High-level system architecture diagrams
 
4. Dev & Ops Support
The bot should assist with:
 
Data extraction for reports
Root-cause analysis using DB logs and queries
Explaining production vs staging differences
Helping new developers onboard faster
Answering questions about deployment and data flows
Expected Bot Capabilities (Examples)
 
✔ “Explain how the daily_summary procedure works”
✔ “Generate an ER diagram for the finance schema”
✔ “What happens if we delete data from this table?”
✔ “Write a query to get monthly exchange closing data”
✔ “Show data flow from ingestion to reporting”
✔ “Create a flow chart of this admin procedure”
 
Tech Stack (Preferred, Flexible)
 
LLM: Gemini
RAG Framework: LangChain / LlamaIndex
Vector DB: Pinecone / Weaviate / FAISS / Chroma
Databases: MySQL
Backend: Python (FastAPI preferred)
Auth: Azure AD / IAM / RBAC
Deployment: Azure (VM)
Diagram Generation: Mermaid / PlantUML
Security & Access
 
Read-only DB access for ingestion
No data used for model training
Private embeddings and enterprise-grade security
On-prem / private cloud deployment only
Deliverables
 
Working RAG-based AI bot (API or UI)
Ingestion pipelines for DB + docs
Query generation and explanation module
Diagram and flow chart generation
Documentation + onboarding guide
Chat Interface

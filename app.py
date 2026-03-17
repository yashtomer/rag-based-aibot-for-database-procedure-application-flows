import streamlit as st
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from src.rag import get_chat_response, get_llm
from src.database import get_full_db_context
from src.vector_store import ingest_schema

# Load env variables
load_dotenv()

st.set_page_config(page_title="DB Intelligence Bot", page_icon="🤖", layout="wide")

st.title("🤖 Database Intelligence Bot")

with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # 1. Check LLM Connectivity
    st.subheader("1. LLM Status")
    if st.button("Check LLM Connectivity"):
        try:
            llm = get_llm()
            # Simple test prompt
            response = llm.invoke("Say the exact word: Connected")
            st.success("✅ LLM Connected successfully!")
        except Exception as e:
            st.error(f"❌ LLM Connection failed: {e}")

    # 2. Check Database Connectivity
    st.subheader("2. Database Status")
    
    def get_base_engine():
        user = os.getenv("MYSQL_USER")
        password = os.getenv("MYSQL_PASSWORD")
        host = os.getenv("MYSQL_HOST")
        port = os.getenv("MYSQL_PORT", "3306")
        if not all([user, host]):
            return None
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/"
        return create_engine(url)

    engine = get_base_engine()
    db_connected = False
    databases = []
    
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            st.success("✅ Database Server Connected!")
            db_connected = True
            
            # Fetch databases
            with engine.connect() as conn:
                result = conn.execute(text("SHOW DATABASES"))
                databases = [row[0] for row in result if row[0] not in ('information_schema', 'mysql', 'performance_schema', 'sys')]
        except Exception as e:
            st.error(f"❌ Database Connection failed: {e}")
    else:
        st.warning("⚠️ Missing Database credentials in .env")

    # 3. Select Database to Ingest
    st.subheader("3. Ingest Database")
    if db_connected and databases:
        selected_db = st.selectbox("Select Database to Ingest", databases)
        if st.button("Ingest and Create Vector DB"):
            with st.spinner(f"Reflecting schema for `{selected_db}` and ingesting..."):
                try:
                    # Set the environment variable so get_full_db_context uses it
                    os.environ["MYSQL_DATABASE"] = selected_db
                    context = get_full_db_context()
                    if context:
                        ingest_schema(context)
                        st.success(f"✅ Successfully ingested `{selected_db}`!")
                    else:
                        st.error("Failed to extract context or schema is empty.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
    elif not db_connected:
        st.info("Connect to database to see available schemas.")
    else:
        st.info("No custom databases found.")

    # 4. Review Table Data
    st.subheader("4. Review Table Data")
    if db_connected and databases:
        review_db = st.selectbox("Select Database to View", databases, key="review_db")
        
        # Get tables
        tables = []
        try:
            target_engine = create_engine(f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT', '3306')}/{review_db}")
            with target_engine.connect() as conn:
                res = conn.execute(text("SHOW TABLES"))
                tables = [row[0] for row in res]
        except Exception as e:
            pass
            
        if tables:
            selected_table = st.selectbox("Select Table", tables)
            if st.button("Review Table Data"):
                try:
                    with target_engine.connect() as conn:
                        df = pd.read_sql(f"SELECT * FROM `{selected_table}` LIMIT 100", conn)
                        st.session_state['review_data'] = df
                        st.session_state['review_table_name'] = selected_table
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
        else:
            st.info("No tables found in this database.")

# Main Chat Area
if 'review_data' in st.session_state:
    st.subheader(f"Data Preview for `{st.session_state['review_table_name']}` (Top 100 rows)")
    st.dataframe(st.session_state['review_data'])
    if st.button("Close Data View"):
        del st.session_state['review_data']
        st.rerun()
    st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Database Intelligence Bot. Ask me anything about your ingested database schema!"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about your database..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from RAG system
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Need to ensure vector db exists, but rag.py handles invoking retrieval
                response_text = get_chat_response(prompt)
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"Error generating response: {e}")

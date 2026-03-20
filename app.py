import streamlit as st
import os
import urllib.parse
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from src.rag import get_chat_response, get_llm
from src.database import get_full_db_context
from src.vector_store import ingest_schema

# Load env variables
load_dotenv()

st.set_page_config(page_title="DB Intelligence Bot", page_icon="🤖", layout="wide")

# Theme state management
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

# ---------------------------------------------------------------------------
# Theme CSS — comprehensive overrides for EVERY Streamlit widget
# ---------------------------------------------------------------------------
_DARK_BG       = "#0E1117"
_DARK_BG2      = "#161b22"
_DARK_SIDEBAR  = "#1a1c24"
_DARK_CARD     = "#21262d"
_DARK_INPUT    = "#262730"
_DARK_BORDER   = "#4a4d5a"
_DARK_TEXT     = "#FAFAFA"
_ACCENT        = "#ff4b4b"

if st.session_state.theme == 'Dark':
    st.markdown(f"""
        <style>
        /* ===== ROOT / APP SHELL ===== */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        .main .block-container {{
            background-color: {_DARK_BG} !important;
            color: {_DARK_TEXT} !important;
        }}

        [data-testid="stHeader"] {{
            background-color: rgba(14, 17, 23, 0.9) !important;
        }}
        [data-testid="stToolbar"] {{
            color: {_DARK_TEXT} !important;
        }}

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {{
            background-color: {_DARK_SIDEBAR} !important;
            color: {_DARK_TEXT} !important;
        }}

        /* ===== BOTTOM BAR / CHAT INPUT ===== */
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stBottom"] > div > div,
        [data-testid="stBottom"] > div > div > div,
        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottomBlockContainer"] > div,
        .stChatInput,
        [data-testid="stChatInput"],
        [data-testid="stChatInputContainer"],
        .stChatInputContainer,
        .stChatInput > div {{
            background-color: {_DARK_BG} !important;
        }}
        [data-testid="stChatInput"] textarea,
        .stChatInput textarea {{
            background-color: {_DARK_INPUT} !important;
            color: {_DARK_TEXT} !important;
            border-color: {_DARK_BORDER} !important;
        }}
        /* Chat input send button area */
        [data-testid="stChatInput"] button,
        .stChatInput button {{
            background-color: {_DARK_INPUT} !important;
            color: {_DARK_TEXT} !important;
        }}

        /* ===== CHAT MESSAGES ===== */
        [data-testid="stChatMessage"] {{
            background-color: {_DARK_CARD} !important;
            border: 1px solid {_DARK_BORDER};
            border-radius: 10px;
        }}

        /* ===== BUTTONS ===== */
        .stButton > button {{
            background-color: {_DARK_INPUT} !important;
            color: {_DARK_TEXT} !important;
            border: 1px solid {_DARK_BORDER} !important;
            transition: all 0.2s ease-in-out;
        }}
        .stButton > button:hover {{
            border-color: {_ACCENT} !important;
            color: {_ACCENT} !important;
        }}

        /* ===== SELECTBOX / DROPDOWNS ===== */
        .stSelectbox div[data-baseweb="select"],
        .stSelectbox div[data-baseweb="select"] > div,
        .stSelectbox [data-baseweb="popover"],
        div[data-baseweb="select"] > div {{
            background-color: {_DARK_INPUT} !important;
            color: {_DARK_TEXT} !important;
            border-color: {_DARK_BORDER} !important;
        }}
        /* Placeholder text inside selectbox — Streamlit uses auto-generated
           st-* classes for the placeholder div, so we must use a broad wildcard */
        .stSelectbox div[data-baseweb="select"] div,
        .stSelectbox div[data-baseweb="select"] span,
        .stSelectbox div[data-baseweb="select"] input,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] span {{
            color: {_DARK_TEXT} !important;
            -webkit-text-fill-color: {_DARK_TEXT} !important;
        }}
        /* Dropdown list options */
        div[data-baseweb="popover"],
        ul[data-testid="stSelectboxVirtualDropdown"],
        div[data-baseweb="menu"],
        div[data-baseweb="popover"] li,
        [data-baseweb="menu"] li {{
            background-color: {_DARK_CARD} !important;
            color: {_DARK_TEXT} !important;
        }}
        div[data-baseweb="popover"] li:hover,
        [data-baseweb="menu"] li:hover {{
            background-color: {_DARK_INPUT} !important;
        }}

        /* ===== TEXT INPUTS / TEXT AREAS ===== */
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {{
            background-color: {_DARK_INPUT} !important;
            color: {_DARK_TEXT} !important;
            border-color: {_DARK_BORDER} !important;
        }}

        /* ===== ALL TEXT ===== */
        .stMarkdown, .stMarkdown p,
        p, span, label,
        h1, h2, h3, h4, h5, h6,
        [data-testid="stWidgetLabel"] label,
        .stRadio label,
        .stCheckbox label {{
            color: {_DARK_TEXT} !important;
        }}

        /* ===== ALERT BOXES (info, success, warning, error) ===== */
        [data-testid="stAlert"],
        .stAlert,
        div[data-baseweb="notification"] {{
            background-color: {_DARK_CARD} !important;
            color: {_DARK_TEXT} !important;
            border-color: {_DARK_BORDER} !important;
        }}

        /* ===== CODE BLOCKS ===== */
        .stCodeBlock, pre, code {{
            background-color: {_DARK_CARD} !important;
            color: {_DARK_TEXT} !important;
        }}

        /* ===== DATAFRAME / TABLES ===== */
        [data-testid="stDataFrame"],
        .stDataFrame {{
            background-color: {_DARK_SIDEBAR} !important;
        }}

        /* ===== EXPANDER ===== */
        details[data-testid="stExpander"] {{
            background-color: {_DARK_CARD} !important;
            border-color: {_DARK_BORDER} !important;
        }}
        details[data-testid="stExpander"] summary {{
            color: {_DARK_TEXT} !important;
        }}

        /* ===== DIVIDER / HR ===== */
        hr {{
            border-color: {_DARK_BORDER} !important;
        }}

        /* ===== SPINNER ===== */
        .stSpinner > div {{
            border-top-color: {_ACCENT} !important;
        }}

        /* ===== TOGGLE / CHECKBOX ===== */
        [data-testid="stToggle"] label span {{
            color: {_DARK_TEXT} !important;
        }}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        /* Light Mode — mostly defaults, just polish */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF;
            color: #262730;
        }
        [data-testid="stHeader"] {
            background-color: rgba(255, 255, 255, 0.9) !important;
        }
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            background-color: #f0f2f6;
        }
        .stButton > button {
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover {
            border-color: #ff4b4b;
            color: #ff4b4b;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Branded Header with Aeologic Logo ---
import base64

def get_logo_base64():
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "aeologo.png")
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

_logo_b64 = get_logo_base64()
if _logo_b64:
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 10px;">
            <img src="data:image/png;base64,{_logo_b64}" alt="Aeologic Logo" style="height: 48px;">
            <h1 style="margin: 0; padding: 0; font-size: 2rem;">Database Intelligence Bot</h1>
        </div>
    """, unsafe_allow_html=True)
else:
    st.title("🤖 Database Intelligence Bot")

with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Theme Toggle UI
    dark_mode = st.toggle("🌓 Dark Mode", value=(st.session_state.theme == "Dark"))
    if dark_mode and st.session_state.theme != "Dark":
        st.session_state.theme = "Dark"
        st.rerun()
    elif not dark_mode and st.session_state.theme != "Light":
        st.session_state.theme = "Light"
        st.rerun()
    # 1. Check LLM Connectivity
    st.subheader("1. LLM Status")
    try:
        llm = get_llm()
        # Use .model attribute which is passed in constructor
        model_display = getattr(llm, 'model', 'gemini-flash-latest')
        st.info(f"Selected Model: `{model_display}`")
    except Exception as e:
        st.error(f"Error identifying LLM: {e}")

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
        user = os.getenv("MYSQL_USER", "").strip('\"\'')
        password = os.getenv("MYSQL_PASSWORD", "").strip('\"\'')
        host = os.getenv("MYSQL_HOST", "").strip('\"\'')
        port = os.getenv("MYSQL_PORT", "3306").strip('\"\'')

        if not all([user, host]):
            return None
        encoded_password = urllib.parse.quote_plus(password) if password else ""
        url = f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/"
        
        # Force print to Docker daemon logging
        import sys
        print(f"\n--- DEBUG LOGGING ---", file=sys.stderr)
        print(f"Raw Password: {password}", file=sys.stderr)
        print(f"Encoded Password: {encoded_password}", file=sys.stderr)
        print(f"Final URL: mysql+pymysql://{user}:{encoded_password}@{host}:{port}/", file=sys.stderr)
        sys.stderr.flush()
        
        # Test directly rendering it to the user's Streamlit browser so they can see immediately
        st.sidebar.info(f"Targeting Host: {host}:{port}\nEncoded Pass starts with: {password[:3]}...")
        
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
        selected_db = st.selectbox("Select Database to Ingest", databases, index=None, placeholder="Choose a database to ingest...")
        if selected_db:
            if st.button("Ingest and Create Vector DB"):
                with st.spinner(f"Reflecting schema for `{selected_db}` and ingesting..."):
                    try:
                        # Set the environment variable so get_full_db_context uses it
                        os.environ["MYSQL_DATABASE"] = selected_db
                        st.info(f"Extracting schema for database: {selected_db}")
                        context = get_full_db_context()
                        if context and not context.startswith("Error"):
                            st.info(f"Schema extracted ({len(context)} chars). Ingesting into Vector DB...")
                            ingest_schema(context)
                            st.success(f"✅ Successfully ingested `{selected_db}`!")
                        else:
                            st.error(f"Failed to extract context or schema is empty. Context: {context[:500]}...")
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    elif not db_connected:
        st.info("Connect to database to see available schemas.")
    else:
        st.info("No custom databases found.")

    # 4. Review Table Data
    st.subheader("4. Review Table Data")
    if db_connected and databases:
        review_db = st.selectbox("Select Database to View", databases, index=None, placeholder="Choose a database to view data...", key="review_db")
        
        if review_db:
            # Get tables
            tables = []
            try:
                password = os.getenv('MYSQL_PASSWORD', "").strip('\"\'')
                encoded_password = urllib.parse.quote_plus(password) if password else ""
                
                user = os.getenv('MYSQL_USER', "").strip('\"\'')
                host = os.getenv('MYSQL_HOST', "").strip('\"\'')

                url = f"mysql+pymysql://{user}:{encoded_password}@{host}:{os.getenv('MYSQL_PORT', '3306').strip('\"\'')}/{review_db}"
                print(f"Connecting to Target Database: mysql+pymysql://{user}:{encoded_password}@{host}:{os.getenv('MYSQL_PORT', '3306')}/{review_db}")
                target_engine = create_engine(url)
                with target_engine.connect() as conn:
                    res = conn.execute(text("SHOW TABLES"))
                    tables = [row[0] for row in res]
            except Exception as e:
                pass
                
            if tables:
                selected_table = st.selectbox("Select Table", tables, index=None, placeholder="Choose a table...")
                if selected_table:
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

import streamlit as st
import os
import re
import urllib.parse
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from src.rag import get_chat_response, get_diagram_response, get_llm, LLM_PROVIDERS, fetch_models
from src.database import get_full_db_context
from src.vector_store import ingest_schema, fetch_embedding_models

# Load env variables
load_dotenv()

st.set_page_config(page_title="DB Intelligence Bot", page_icon="🤖", layout="wide")


def format_llm_error(e: Exception) -> tuple[str, str]:
    """Parse LLM exceptions into a user-friendly (title, detail) pair."""
    err = str(e)

    # Context length exceeded
    if "context_length_exceeded" in err or "reduce the length" in err.lower():
        return (
            "📏 Context Length Exceeded",
            "The input is too long for the selected model. Try a model with a larger context window or ask a more specific question.",
        )

    # Quota / rate-limit errors
    if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate" in err.lower():
        # Extract retry delay if present
        retry_match = re.search(r"retry(?:\s+in|Delay['\"]:\s*['\"])?\s*([\d.]+)\s*s", err, re.IGNORECASE)
        retry_hint = f"  \nRetry in **{int(float(retry_match.group(1)))} seconds**." if retry_match else ""
        return (
            "⚠️ API Quota Exceeded",
            f"You've hit the rate limit for this model. "
            f"Please try a different model, switch providers, or wait and retry.{retry_hint}",
        )

    # Auth errors
    if "401" in err or "403" in err or "PERMISSION_DENIED" in err or "invalid" in err.lower() and "key" in err.lower():
        return (
            "🔑 Authentication Failed",
            "The API key is invalid or missing permissions. Please check your `.env` file.",
        )

    # Model not found
    if "404" in err or "not found" in err.lower():
        return (
            "🔍 Model Not Found",
            "The selected model is not available. Please choose a different model from the dropdown.",
        )

    # Timeout
    if "timeout" in err.lower():
        return (
            "⏱️ Request Timed Out",
            "The LLM took too long to respond. Please try again.",
        )

    # Generic fallback
    return ("❌ LLM Error", str(e))


def render_mermaid(mermaid_code: str):
    """Render a Mermaid diagram as an image using mermaid.ink."""
    import base64
    encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("ascii")
    img_url = f"https://mermaid.ink/img/{encoded}"
    st.image(img_url, use_container_width=True)
    with st.expander("View Mermaid source"):
        st.code(mermaid_code, language="mermaid")


def render_message(content: str):
    """Render a chat message, handling mermaid code blocks specially."""
    if "```mermaid" in content:
        parts = content.split("```mermaid")
        st.markdown(parts[0])
        for part in parts[1:]:
            if "```" in part:
                mermaid_code, rest = part.split("```", 1)
                render_mermaid(mermaid_code.strip())
                if rest.strip():
                    st.markdown(rest.strip())
            else:
                st.markdown(part)
    else:
        st.markdown(content)


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
    st.subheader("1. LLM Provider")
    _providers = list(LLM_PROVIDERS.keys())
    selected_provider = st.selectbox(
        "Select LLM Provider",
        _providers,
        index=_providers.index("Groq") if "Groq" in _providers else 0,
        key="llm_provider",
    )

    # Fetch models live, cache per provider in session state
    cache_key = f"models_{selected_provider}"
    details_key = f"model_details_{selected_provider}"
    if cache_key not in st.session_state:
        with st.spinner("Fetching available models..."):
            models_list, models_details = fetch_models(selected_provider)
            st.session_state[cache_key] = models_list
            st.session_state[details_key] = models_details

    available_models = st.session_state[cache_key]

    # Default to kimi-k2-instruct-0905 for Groq
    _default_model = "moonshotai/kimi-k2-instruct-0905"
    _model_index = 0
    if selected_provider == "Groq" and _default_model in available_models:
        _model_index = available_models.index(_default_model)

    col_model, col_refresh = st.columns([4, 1])
    with col_model:
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=_model_index,
            key="llm_model",
        )
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄", help="Refresh model list"):
            models_list, models_details = fetch_models(selected_provider)
            st.session_state[cache_key] = models_list
            st.session_state[details_key] = models_details
            st.rerun()

    if st.button("Check LLM Connectivity"):
        try:
            llm = get_llm(selected_provider, selected_model)
            response = llm.invoke("Say the exact word: Connected")
            st.success("✅ LLM Connected successfully!")
        except Exception as e:
            title, detail = format_llm_error(e)
            st.error(f"**{title}**\n\n{detail}")

    st.markdown("---")
    st.subheader("Embedding Model")
    st.caption("Powered by Google (Groq does not offer embedding models)")
    if "embedding_models" not in st.session_state:
        with st.spinner("Fetching embedding models..."):
            st.session_state["embedding_models"] = fetch_embedding_models()

    col_emb, col_emb_refresh = st.columns([4, 1])
    with col_emb:
        selected_embedding = st.selectbox(
            "Select Embedding Model",
            st.session_state["embedding_models"],
            index=0,
            key="embedding_model",
        )
    with col_emb_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄", help="Refresh embedding model list", key="refresh_emb"):
            st.session_state["embedding_models"] = fetch_embedding_models()
            st.rerun()

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
                            ingest_schema(context, st.session_state.get("embedding_model"))
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

                    # Fetch columns and store example queries for main area
                    try:
                        with target_engine.connect() as conn:
                            cols_result = conn.execute(text(f"SHOW COLUMNS FROM `{selected_table}`"))
                            columns = [row[0] for row in cols_result]
                    except Exception:
                        columns = []

                    if columns:
                        example_queries = [
                            f"What are all the columns in the `{selected_table}` table?",
                            f"Describe the structure and purpose of the `{selected_table}` table.",
                            f"Which tables have a relationship with `{selected_table}`?",
                            f"What is the data type of `{columns[0]}` in `{selected_table}`?",
                            f"Show ER diagram for `{selected_table}` and its related tables",
                        ]
                        if len(columns) > 1:
                            example_queries.append(
                                f"How are `{columns[0]}` and `{columns[1]}` used in `{selected_table}`?"
                            )
                        st.session_state["example_queries"] = example_queries
                    else:
                        st.session_state.pop("example_queries", None)
            else:
                st.info("No tables found in this database.")

# Model Details Panel
_provider = st.session_state.get("llm_provider", "Gemini (Google)")
_model = st.session_state.get("llm_model")
_details_key = f"model_details_{_provider}"
_model_info = st.session_state.get(_details_key, {}).get(_model)
if _model_info:
    st.markdown(f"**📋 Model Info: {_model}**")
    ctx = _model_info.get("context_window", "N/A")
    ctx_display = f"{ctx:,}" if isinstance(ctx, int) else str(ctx)
    out = _model_info.get("max_output_tokens", "N/A")
    out_display = f"{out:,}" if isinstance(out, int) else str(out)
    st.markdown(
        f"**Provider:** {_model_info.get('owner', 'N/A')} &nbsp;|&nbsp; "
        f"**Context Window:** {ctx_display} tokens &nbsp;|&nbsp; "
        f"**Max Output:** {out_display} tokens"
    )
    desc = _model_info.get("description", "")
    if desc and desc != "N/A":
        st.caption(desc)
    st.markdown("---")

# Main Chat Area
if 'review_data' in st.session_state:
    st.subheader(f"Data Preview for `{st.session_state['review_table_name']}` (Top 100 rows)")
    st.dataframe(st.session_state['review_data'])
    if st.button("Close Data View"):
        del st.session_state['review_data']
        st.rerun()
    st.markdown("---")

# Example queries section
if "example_queries" in st.session_state:
    st.markdown("**💡 Example queries:**")
    cols = st.columns(2)
    for i, eq in enumerate(st.session_state["example_queries"]):
        with cols[i % 2]:
            if st.button(eq, key=f"eq_{eq}"):
                st.session_state["prefill_query"] = eq
                st.rerun()
    st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Database Intelligence Bot. Ask me anything about your ingested database schema!"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message(message["content"])

# Handle prefilled query from example buttons
prefill = st.session_state.pop("prefill_query", None)

# React to user input
if prompt := (prefill or st.chat_input("Ask about your database...")):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Detect if user is asking for a diagram
    _diagram_keywords = ["er diagram", "erd", "entity relationship", "diagram", "visualize", "schema diagram", "draw", "show diagram", "generate diagram", "relationship diagram"]
    _is_diagram_request = any(kw in prompt.lower() for kw in _diagram_keywords)

    # Get response from RAG system
    with st.chat_message("assistant"):
        with st.spinner("Generating diagram..." if _is_diagram_request else "Thinking..."):
            try:
                _llm_provider = st.session_state.get("llm_provider", "Gemini (Google)")
                _llm_model = st.session_state.get("llm_model")
                _emb_model = st.session_state.get("embedding_model")

                if _is_diagram_request:
                    response_text = get_diagram_response(prompt, _llm_provider, _llm_model, _emb_model)
                    # Extract mermaid code from response
                    mermaid_code = response_text
                    if "```mermaid" in mermaid_code:
                        mermaid_code = mermaid_code.split("```mermaid")[1].split("```")[0].strip()
                    elif "```" in mermaid_code:
                        mermaid_code = mermaid_code.split("```")[1].split("```")[0].strip()

                    # Render the Mermaid diagram
                    st.markdown("Here's the ER diagram for your database:")
                    render_mermaid(mermaid_code)

                    # Store both text and mermaid code for history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Here's the ER diagram for your database:\n\n```mermaid\n{mermaid_code}\n```",
                    })
                else:
                    response_text = get_chat_response(prompt, _llm_provider, _llm_model, _emb_model)
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                title, detail = format_llm_error(e)
                st.error(f"**{title}**\n\n{detail}")

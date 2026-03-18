import os
import urllib.parse
from typing import List
from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv

load_dotenv()

def get_db_url() -> str:
    user = os.getenv("MYSQL_USER", "").strip('\"\'')
    password = os.getenv("MYSQL_PASSWORD", "").strip('\"\'')
    host = os.getenv("MYSQL_HOST", "").strip('\"\'')
    port = os.getenv("MYSQL_PORT", "3306").strip('\"\'')
    db = os.getenv("MYSQL_DATABASE", "").strip('\"\'')

    if not all([user, host, db]):
        print("Warning: Missing database configuration in .env. Using a mock SQLite DB for demonstration.")
        return "sqlite:///:memory:"

    encoded_password = urllib.parse.quote_plus(password) if password else ""
    db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
    print(f"Connecting to Full Database: mysql+pymysql://{user}:{password}@{host}:{port}/{db}")
    print(f"Password Being Used: {password}")
    return db_url

def get_engine():
    return create_engine(get_db_url())

def get_schema_summary(engine) -> str:
    """
    Introspects the database and returns a text summary of the schema (tables, columns, relationships).
    """
    inspector = inspect(engine)
    schema_texts = []

    try:
        table_names = inspector.get_table_names()
    except Exception as e:
        return f"Error retrieving table names: {e}"
    #print(table_names)
    for table_name in table_names:
        table_summary = [f"Table: {table_name}"]
        #print(table_summary)
        # Columns
        columns = []
        try:
            for col in inspector.get_columns(table_name):
                col_type = str(col['type'])
                col_str = f"- {col['name']} ({col_type})"
                if col.get('primary_key'):
                    col_str += " [PK]"
                if col.get('nullable'):
                    col_str += " [NULL]"
                columns.append(col_str)
            table_summary.append("Columns:\n" + "\n".join(columns))
        except Exception as e:
             table_summary.append(f"Error reading columns: {e}")
        #print("table_summary final",table_summary)
        # Foreign Keys
        try:
            fks = inspector.get_foreign_keys(table_name)
            if fks:
                fk_list = []
                for fk in fks:
                    constrained = ", ".join(fk['constrained_columns'])
                    referred = ", ".join(fk['referred_columns'])
                    fk_list.append(f"- {constrained} -> {fk['referred_table']}.{referred}")
                table_summary.append("Foreign Keys:\n" + "\n".join(fk_list))
        except Exception as e:
            pass

        schema_texts.append("\n".join(table_summary))

    return "\n\n".join(schema_texts)

def get_stored_procedures(engine) -> str:
    """
    Fetches stored procedures and their definitions from MySQL.
    """
    # This query is specific to MySQL
    if engine.dialect.name != "mysql":
        return ""

    db_name = os.getenv("MYSQL_DATABASE")
    if not db_name:
        return ""

    query = text("""
        SELECT ROUTINE_NAME, ROUTINE_DEFINITION
        FROM information_schema.ROUTINES
        WHERE ROUTINE_SCHEMA = :db_name AND ROUTINE_TYPE = 'PROCEDURE'
    """)

    procedures_text = []
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"db_name": db_name})
            for row in result:
                name = row[0]
                # Routine definition might be None if user doesn't have permissions or it's not a viewable routine
                definition = row[1] if row[1] else "No definition available"
                procedures_text.append(f"Stored Procedure: {name}\nDefinition:\n{definition}")
    except Exception as e:
        print(f"Error fetching stored procedures: {e}")
        return ""

    return "\n\n".join(procedures_text)

def get_full_db_context() -> str:
    """
    Combines schema summary and stored procedures into a single context string.
    """
    engine = get_engine()
    schema = get_schema_summary(engine)
    procs = get_stored_procedures(engine)
   
    return f"DATABASE SCHEMA:\n{schema}\n\nSTORED PROCEDURES:\n{procs}"

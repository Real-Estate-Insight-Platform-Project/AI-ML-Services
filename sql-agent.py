import os
import re
import time
from typing import Dict, Any, List, Optional

# Optional: load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import streamlit as st
from sqlalchemy import create_engine, event, text

# LangChain
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentType, create_sql_agent
from langchain import hub


# Configuration / Env


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY is not set. Set it in your environment or .env file.")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    PG_HOST = os.getenv("PG_HOST")
    PG_PORT = os.getenv("PG_PORT")  # Supabase pooler default
    PG_DB = os.getenv("PG_DB")
    PG_USER = os.getenv("PG_USER")
    PG_PASSWORD = os.getenv("PG_PASSWORD")
    PG_SSLMODE = os.getenv("PG_SSLMODE")
    DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode={PG_SSLMODE}"

# Allowlist exactly like the notebook
ALLOWED_TABLES = ["properties", "market_analytics"]

# Debug logging of executed SQL
LOG_SQL = os.getenv("LOG_SQL", "false").lower() == "true"


# SQLAlchemy engine (read-only + timeouts) 


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
    connect_args={
        # server-side protection against runaway queries (5s)
        "options": "-c statement_timeout=5000 -c idle_in_transaction_session_timeout=5000"
    },
)

@event.listens_for(engine, "connect")
def _enforce_readonly(dbapi_conn, record):
    # Optional belt & suspenders: force read-only transactions
    with dbapi_conn.cursor() as c:
        c.execute("SET default_transaction_read_only = on;")

# LangChain DB wrapper (ALLOWLIST tables)


db = SQLDatabase(engine=engine, include_tables=ALLOWED_TABLES)

# Deterministic LLM

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Toolkit + runtime SQL guardrails

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

_original_db_run = db.run
_limit_re = re.compile(r"\\bLIMIT\\s+\\d+", re.IGNORECASE)
_last_sql: Optional[str] = None  # capture last executed SQL for UI preview

def _guarded_run(sql: str, *args, **kwargs):
    global _last_sql
    _last_sql = sql  # record what the agent is about to run

    up = sql.strip().upper()
    if not up.startswith("SELECT"):
        raise ValueError("Only SELECT queries are permitted.")
    if _limit_re.search(sql) is None:
        sql = sql.rstrip().rstrip(";") + " LIMIT 100"
    if LOG_SQL:
        print("\\n--- SQL EXECUTED ---\\n", sql)
    return _original_db_run(sql, *args, **kwargs)

db.run = _guarded_run  # enforce guardrails


# Prompt from Hub + inject live schema snippet


try:
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    base_system = prompt_template.format(dialect="PostgreSQL", top_k=5)
except Exception as e:
    # Fallback if Hub fetch fails (e.g., offline)
    base_system = (
        "You are a helpful SQL assistant for PostgreSQL. Use ONLY SELECT statements. "
        "Return succinct answers. Prefer aggregates and always include a LIMIT (<=100) unless a single value is asked."
    )

schema_snippet = db.get_table_info(ALLOWED_TABLES)

merged_system_prompt = (
    f"{base_system}\\n\\n"
    "### ADDITIONAL CONTEXT: DATABASE SCHEMA (READ CAREFULLY)\\n"
    "Use ONLY these tables/columns. Prefer aggregates. Always include a LIMIT (<=100) unless a single value is asked.\\n\\n"
    f"{schema_snippet}"
)

# Create SQL Agent

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.OPENAI_FUNCTIONS,  # robust tool-calling
    verbose=False,
    prefix=merged_system_prompt,
)

def ask(question: str) -> str:
    """Natural language → (SQL) → Answer using merged Hub+Schema prompt."""
    return agent.run(question)


# Streamlit UI


st.set_page_config(page_title="SQL Agent", page_icon="🧠", layout="wide")
st.title("🧠 SQL Agent")

with st.expander("Connection health", expanded=False):
    try:
        with engine.connect() as conn:
            who = conn.execute(text("select current_user")).scalar()
            props = conn.execute(text("select count(*) from public.properties")).scalar()
            marts = conn.execute(text("select count(*) from public.market_analytics")).scalar()
        st.success(f"Connected as **{who}** · properties={props} · market_analytics={marts}")
    except Exception as e:
        st.error(f"DB connection error: {e}")

st.markdown("""
**Notes**
- Uses *only* tables: `public.properties`, `public.market_analytics`
- Guardrails: **SELECT-only**, auto-**LIMIT 100** if missing, **read-only** session
- Model: `gpt-4o-mini` (temperature 0)
""")

default_q = "List all active properties in Austin with their price and number of bedrooms."
q = st.text_area("Your question", value=default_q, height=100, placeholder="Ask something about properties or market analytics…")
btn = st.button("Ask")

if btn:
    t0 = time.perf_counter()
    try:
        answer = ask(q.strip())
        elapsed = time.perf_counter() - t0
        st.subheader("Answer")
        st.write(answer)

        st.subheader("SQL executed")
        st.code(_last_sql or "(not captured)", language="sql")

        # Optional: show preview rows by running the captured SQL again
        if _last_sql:
            try:
                with engine.connect() as conn:
                    # Respect guardrail LIMIT (already injected)
                    res = conn.execute(text(_last_sql))
                    rows = res.fetchall()
                    cols = res.keys()
                if rows:
                    import pandas as pd  # local import to keep import list light
                    df = pd.DataFrame(rows, columns=cols)
                    st.subheader("Rows preview")
                    st.dataframe(df, use_container_width=True, height=300)
                else:
                    st.info("Query returned 0 rows.")
            except Exception as ex:
                st.warning(f"Could not preview rows: {ex}")
        st.caption(f"Elapsed: {elapsed:.2f}s")
    except Exception as e:
        st.error(f"""Agent error: {e} 
        
Tips:
- Ensure RLS allows your role to read (either grant `authenticated` to `sql_agent` **or** target RLS to `reporting`).
- Ensure DATABASE_URL points to your Supabase pooler (port 6543) with sslmode=require.
""")


if __name__ == "__main__":
    # For direct execution (optional): print a small hint.
    print("Run the Streamlit UI with:  streamlit run sql-agent.py")

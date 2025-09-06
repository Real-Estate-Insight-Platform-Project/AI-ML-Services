import os, re
from dotenv import load_dotenv, find_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, create_sql_agent
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sqlalchemy import create_engine, event
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv(find_dotenv(), override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
LOG_SQL = (os.getenv("LOG_SQL", "false").lower() == "true")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# SQLAlchemy engine with timeouts & read-only transactions
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
    connect_args={
        "options": "-c statement_timeout=5000 -c idle_in_transaction_session_timeout=5000"
    },
)

@event.listens_for(engine, "connect")
def _enforce_readonly(dbapi_conn, record):
    with dbapi_conn.cursor() as c:
        c.execute("SET default_transaction_read_only = on;")

# LangChain DB wrapper (ALLOWLIST tables for safety)
ALLOWED_TABLES = ["properties", "market_analytics"]
db = SQLDatabase(engine=engine, include_tables=ALLOWED_TABLES)

# Deterministic LLM for reliable SQL generation
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
)

# Toolkit + runtime SQL guardrails (SELECT-only + LIMIT)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
_original_db_run = db.run
_limit_re = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)

def _guarded_run(sql: str, *args, **kwargs):
    up = sql.strip().upper()
    if not up.startswith("SELECT"):
        raise ValueError("Only SELECT queries are permitted.")
    if not _limit_re.search(sql):
        sql += " LIMIT 100"
    return _original_db_run(sql, *args, **kwargs)

db.run = _guarded_run

if LOG_SQL:
    def _logging_run(sql: str, *a, **k):
        print("\n--- SQL EXECUTED ---\n", sql)
        return _guarded_run(sql, *a, **k)
    db.run = _logging_run

# Pull LangChain Hub prompt AND inject live schema snippet
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
base_system = prompt_template.format(dialect="PostgreSQL", top_k=5)
schema_snippet = db.get_table_info(ALLOWED_TABLES)
merged_system_prompt = (
    f"{base_system}\n\n"
    "### ADDITIONAL CONTEXT: DATABASE SCHEMA (READ CAREFULLY)\n"
    "Use ONLY these tables/columns. Prefer aggregates. Always include a LIMIT (<=100) unless a single value is asked.\n\n"
    f"{schema_snippet}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", merged_system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_sql_agent(
    llm,
    db=db,
    prompt=prompt,
    agent_type="tool-calling",
    verbose=True,
    top_k=5,
    max_iterations=15,
)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_endpoint(request: QueryRequest):
    try:
        answer = agent.run(request.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "SQL Agent API is running."}

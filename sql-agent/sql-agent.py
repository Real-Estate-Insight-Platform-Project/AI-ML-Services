import os, re
from dotenv import load_dotenv, find_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType
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
# UPDATED: allow only these four tables (exclude 'profiles' and 'user_favorites')
ALLOWED_TABLES = ["nri_counties", "predictions", "properties", "state_market"]
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


# Pull LangChain Hub prompt AND inject live schema snippet + domain notes
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
base_system = prompt_template.format(dialect="PostgreSQL", top_k=5)
schema_snippet = db.get_table_info(ALLOWED_TABLES)

domain_notes = """
### DOMAIN NOTES (READ FIRST)
- You are querying a read-only PostgreSQL database. Only SELECT statements are allowed.
- **ROW LIMIT POLICY:** For any query that can return multiple rows (e.g., property listings or historical rows), **always** include `LIMIT 10` (hard cap). 
  - If the user requests more than 10 rows, still return only the first 10 and state that results are truncated to 10.
  - Do **not** apply a LIMIT when the question clearly asks for a single value/aggregate (e.g., COUNT, AVG, MIN/MAX for a specific filter).
- Use ONLY these allowed tables: `nri_counties`, `predictions`, `properties`, `state_market`. Do NOT use `profiles` or `user_favorites`.
- Prefer aggregates for summaries and apply precise WHERE filters (by state/year/month/price/bedrooms) from the user’s question.

### TABLE PURPOSES & KEYS
1) public.predictions — (FORECASTS)
   - PK: (year, month, state).
   - Columns: median_listing_price, average_listing_price, median_listing_price_per_square_foot, total_listing_count, median_days_on_market, market_trend.
   - **BUSINESS RULE:** If “predictions” are requested without dates, return **current month and next two months** (3-month horizon) for the specified state(s).
     Compute “current month” as `(extract(year from now()), extract(month from now()))`. Sort by (year, month) ASC. The 3-month horizon is already small; no LIMIT needed unless returning multiple states (then still apply LIMIT 10 to the row set).

2) public.state_market — (HISTORICAL OBSERVATIONS)
   - PK: (year, month, state).
   - Use for historical trends, M/M and Y/Y comparisons, and time series. 
   - When returning row-level history (e.g., many months or many states), **enforce `LIMIT 10`** and sort by `year DESC, month DESC` unless the user specifies an order.

3) public.nri_counties — (RISK INDEX, COUNTY-LEVEL)
   - PK: county_fips.
   - risk_index_score is a 0–100 **percentile** (relative within county level). risk_index_rating is a qualitative bucket (“Very Low” … “Very High”).
   - `predominant_hazard` is derived from the **highest Expected Annual Loss (EAL) total** among 18 hazards.

4) public.properties — (LISTINGS CATALOG)
   - PK: id (uuid). Columns include price, bedrooms, bathrooms, square_feet, property_type, listing_status, city/state/zip, created_at/updated_at, coordinates.
   - For listing queries (e.g., “3+ bedrooms under $500k”), **always** `LIMIT 10` and default ordering `ORDER BY created_at DESC NULLS LAST` unless the user asks for a different order (e.g., price ASC).

### JOIN & FILTER HINTS
- State-level joins: `predictions.state` <-> `state_market.state` (text). County risk lives in `nri_counties`; aggregate by state via `state_name`/`state_fips` if needed.
- Time filters use (year, month) pairs; sort by `year, month`.
- When combining historical (state_market) with forecasts (predictions), show recent history first (limited to 10 rows if multi-row), then the 3-month forecast window.

### ANSWER STYLE
- Clearly label **historical** (state_market) vs **predicted** (predictions).
- Include units (currency for prices, counts as integers, ratios as % when appropriate).
- If truncation occurs due to the 10-row cap, **say so explicitly** (e.g., “showing first 10 rows”).
"""

merged_system_prompt = (
    f"{base_system}\n\n"
    "### ADDITIONAL CONTEXT: DATABASE SCHEMA (READ CAREFULLY)\n"
    "Use ONLY these tables/columns. Prefer aggregates. For multi-row outputs, always include LIMIT 10 (hard cap) unless a single value is asked.\n\n"
    f"{schema_snippet}\n\n"
    f"{domain_notes}"
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

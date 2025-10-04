import os, re, math
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
from geoalchemy2 import Geometry

# Load environment variables
load_dotenv(find_dotenv(), override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("SQL_AGENT_DATABASE_URL")
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
    # Keep your existing LIMIT safety; presentation layer will still cap to 10.
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
   - PK: id (uuid). Columns include price, bedrooms, bathrooms, square_feet, property_type, listing_status, address, city, state, created_at/updated_at.
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

# --------------------------
# Pretty-printing utilities
# --------------------------

def _parse_markdown_table(md: str):
    """
    Parse a simple Markdown table into list[dict]. 
    Expects header row and '---' separator. Ignores empty/short lines.
    """
    lines = [ln for ln in md.splitlines() if ln.strip()]
    # Find the first header row that looks like a pipe table
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|") and ("|" in ln.strip()[1:]):
            # Next line should be the separator (---)
            if i + 1 < len(lines) and set(lines[i + 1].replace("|", "").replace(" ", "")) <= set("-"):
                start = i
                break
    if start is None:
        return None

    header = [h.strip() for h in lines[start].split("|") if h.strip()]
    rows = []
    for ln in lines[start + 2:]:
        if not ln.strip().startswith("|"):
            # stop at first non-table block
            break
        cols = [c.strip() for c in ln.split("|") if c.strip()]
        if len(cols) < 1:
            continue
        row = {}
        for j, h in enumerate(header):
            if j < len(cols):
                row[h] = cols[j]
        rows.append(row)
    return rows

def _coerce_number(x):
    try:
        # remove $ and commas
        s = str(x).replace("$", "").replace(",", "").strip()
        if s == "" or s.lower() == "n/a":
            return None
        return float(s)
    except Exception:
        return None

EM_SPACE = " "  # U+2003 for wide gap

def format_properties(rows):
    """
    rows: list[dict] with keys like Title, Price, Bedrooms, Bathrooms, Address, square_feet, etc.
    Returns a numbered, card-like Markdown string (without IDs).
    Caps to first 10 entries.
    """
    if not rows:
        return None

    # Normalize field names (case-insensitive)
    def g(row, *keys, default=""):
        for k in keys:
            for kk in row.keys():
                if kk.lower() == k.lower():
                    return row[kk]
        return default

    cards = []
    for idx, row in enumerate(rows[:10], start=1):
        title = g(row, "Title", "title", default="Unnamed Property")
        price_raw = g(row, "Price", "price", default="")
        price_num = _coerce_number(price_raw)
        price = f"${price_num:,.0f}" if price_num is not None else (price_raw or "N/A")

        bedrooms = g(row, "Bedrooms", "bedrooms", default="")
        bathrooms = g(row, "Bathrooms", "bathrooms", default="")
        sqft = g(row, "Square Feet", "square_feet", "sqft", default="")

        # --- Location fields (robust to header variants) ---
        address = g(
            row,
            "address", "street_address", "street address", "addr", "full_address", "location",
            default=""
        )
        city = g(row, "city", "town", "locality", default="")
        state = g(row, "state", "state_code", "region", "province", default="")

        # Build the card (address + city + state together)
        location_line = ""
        if address or city or state:
            parts = []
            if address: parts.append(address)
            if city:    parts.append(city)
            if state:   parts.append(state)
            location_line = f"📍 {', '.join(parts)}\n"

        # Build the meta info
        meta_line = f"🛏 {bedrooms} bed"
        if bathrooms:
            meta_line += f"{EM_SPACE*3}🛁 {bathrooms} bath"
        if sqft:
            sqn = _coerce_number(sqft)
            sqft_disp = f"{int(sqn):,}" if (sqn is not None and not math.isnan(sqn)) else str(sqft)
            meta_line += f"{EM_SPACE*3}📐 {sqft_disp} sqft"

        card = (
            f"{idx}) {title}\n"
            + location_line
            + meta_line + "\n"
            + f"💰 {price}\n"
        )
        cards.append(card)

    return "\n\n".join(cards)

def try_beautify_properties(answer_text: str):
    """
    If the agent returned a markdown table that looks like property rows,
    format them as numbered cards without IDs.
    """
    rows = _parse_markdown_table(answer_text)
    if not rows:
        return None

    # Heuristic: if the table has a Title/Price column, it's likely properties.
    header_keys = {k.lower() for k in rows[0].keys()}
    if not (("title" in header_keys) and ("price" in header_keys)):
        return None

    # Drop obvious ID fields from rows (not shown to users)
    for r in rows:
        for k in list(r.keys()):
            if k.strip().lower() in {"id", "uuid"}:
                r.pop(k, None)

    pretty = format_properties(rows)
    return pretty


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

        # Try to beautify property results into card style (numbered, spaced)
        pretty = try_beautify_properties(answer)
        if pretty:
            # Mention truncation if we likely capped to 10
            answer = "Here are the top matching properties (showing up to 10):\n\n" + pretty

        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "SQL Agent API is running."}

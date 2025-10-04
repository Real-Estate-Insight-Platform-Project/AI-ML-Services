# main.py
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
    connect_args={"options": "-c statement_timeout=5000 -c idle_in_transaction_session_timeout=5000"},
)

@event.listens_for(engine, "connect")
def _enforce_readonly(dbapi_conn, record):
    with dbapi_conn.cursor() as c:
        c.execute("SET default_transaction_read_only = on;")

# LangChain DB wrapper (ALLOWLIST tables for safety)
# allow only these four tables (exclude 'profiles' and 'user_favorites')
ALLOWED_TABLES = ["nri_counties", "predictions", "properties", "state_market"]
db = SQLDatabase(engine=engine, include_tables=ALLOWED_TABLES)

# Deterministic LLM for reliable SQL generation
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

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
- You query a read-only PostgreSQL. Only SELECT is allowed. If the user asks for non-read ops, refuse.
- ROW LIMIT POLICY:
  - For multi-row outputs (lists, time series, states, properties), ALWAYS include `LIMIT 10` (hard cap).
  - If the user asks for more, still cap to 10 and state that results are truncated.
  - Do NOT force a LIMIT when the answer is a single scalar/aggregate (COUNT/AVG/MIN/MAX for a specific filter).
- Allowed tables only: `nri_counties`, `predictions`, `properties`, `state_market`.
  - Never access or mention user/agent tables. Never expose emails, phone numbers, account IDs, or any PII.

### TIME / PREDICTION RULES
- Current year/month in SQL: `extract(year from now())::int`, `extract(month from now())::int`.
- If the user requests “predictions” without an explicit year/month:
  - Return **current month plus the next two months** (3-month horizon) from `predictions`.
  - Sort by (year, month) ASC.
- “Which states are predicted to have price growth this September?”:
  - Interpret “price growth” as `market_trend = 'rising'` for the specified (year, month).
  - If the user omits year/month, infer from `now()`.

### TABLE PURPOSES
1) public.predictions — FORECASTS at state–month grain. PK = (year, month, state).
   Columns include: `median_listing_price`, `average_listing_price`,
   `median_listing_price_per_square_foot`, `total_listing_count`,
   `median_days_on_market`, `market_trend`.
2) public.state_market — HISTORICAL state–month observations used for MoM/YoY and long trends.
   For multi-row outputs: sort by `year DESC, month DESC`, LIMIT 10.
3) public.nri_counties — FEMA NRI at county level. Use `AVG(risk_index_score)` grouped by `state_name` for state risk comparisons.
4) public.properties — Listings catalog. Default ordering for listings: `ORDER BY created_at DESC NULLS LAST` unless the user asks otherwise.

### HOW TO ANSWER COMMON QUESTIONS
- “States where properties sell faster than 1 week”:
  - Use the **state-level prediction** for the month in question (default = current month) and filter on `predictions.median_days_on_market < 7`.
  - Do NOT ask for property-level sale data to answer this.
- “Which states are predicted to have price growth this September?”:
  - `SELECT state FROM predictions WHERE year=? AND month=? AND market_trend='rising' ORDER BY state LIMIT 10;`
- When mixing historical vs predicted, label clearly which table each number comes from.

### JOIN & FILTER HINTS
- State-level joins via text `state`. Risk comparisons: aggregate `nri_counties` by `state_name`.
- Prefer precise WHERE filters by state/year/month/price/bed/bath.

### ANSWER STYLE
- Label **Historical (state_market)** vs **Predicted (predictions)** when both appear.
- Show units: currency ($) for prices, integers for counts, and % for ratios when appropriate.
- If truncated to 10 rows, explicitly say “showing first 10 rows”.

### PROPERTY CARD FORMAT (STRICT)
When returning **individual properties**, format EACH property EXACTLY as numbered list of properties , with four lines for each property, like this:
   {{Title}}
   {{Address}}, {{City}}, {{State}}
   🛏 {{Bedrooms}} bed   🛁 {{Bathrooms}} bath   📐 {{SquareFeet}} sqft
   💰 ${{Price}}

- No IDs, emails, phone numbers, or internal fields.
- Show up to 10 properties max.
- If the query would produce more than 10 properties, return the first 10 and say results are truncated.


### EXAMPLES (imitate these)
-- Current month + next 2 months forecasts for California
SELECT year, month, state, median_listing_price, median_days_on_market, market_trend
FROM predictions
WHERE state = 'California'
  AND (year > extract(year from now())::int
       OR (year = extract(year from now())::int AND month >= extract(month from now())::int))
ORDER BY year, month
LIMIT 10;

-- States selling faster than 1 week this month
SELECT state, median_days_on_market
FROM predictions
WHERE year  = extract(year from now())::int
  AND month = extract(month from now())::int
  AND median_days_on_market < 7
ORDER BY median_days_on_market ASC
LIMIT 10;

-- “Price growth this September”
SELECT state
FROM predictions
WHERE year = 2025 AND month = 9 AND market_trend = 'rising'
ORDER BY state
LIMIT 10;

-- Risk-adjusted opportunity (low risk + rising)
WITH risk AS (
  SELECT state_name AS state, AVG(risk_index_score) AS avg_risk
  FROM nri_counties
  GROUP BY state_name
)
SELECT p.state, p.market_trend, r.avg_risk, p.median_listing_price
FROM predictions p
JOIN risk r ON r.state = p.state
WHERE p.year = extract(year from now())::int
  AND p.month = extract(month from now())::int
  AND p.market_trend = 'rising'
  AND r.avg_risk <= 40
ORDER BY r.avg_risk ASC, p.median_listing_price ASC
LIMIT 10;

### PRIVACY & SAFETY
- Never attempt to query or reveal emails, phone numbers, account identifiers, or any agent PII.
- Only SELECTs are allowed. Refuse any non-SELECT request.
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
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|") and ("|" in ln.strip()[1:]):
            if i + 1 < len(lines) and set(lines[i + 1].replace("|", "").replace(" ", "")) <= set("-"):
                start = i
                break
    if start is None:
        return None

    header = [h.strip() for h in lines[start].split("|") if h.strip()]
    rows = []
    for ln in lines[start + 2:]:
        if not ln.strip().startswith("|"):
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
        s = str(x).replace("$", "").replace(",", "").strip()
        if s == "" or s.lower() == "n/a":
            return None
        return float(s)
    except Exception:
        return None

EM_SPACE = " "  # U+2003 for wide gap

def format_properties(rows):
    """
    rows: list[dict] with keys like Title, Price, Bedrooms, Bathrooms, Address, square_feet, City, State, etc.
    Returns a numbered, card-like Markdown string (without IDs), in the STRICT four-line format.
    Caps to first 10 entries.
    """
    if not rows:
        return None

    def g(row, *keys, default=""):
        for k in keys:
            for kk in row.keys():
                if kk.lower() == k.lower():
                    return row[kk]
        return default

    cards = []
    for idx, row in enumerate(rows[:10], start=1):
        title = g(row, "Title", "title", default="Unnamed Property")

        # Location
        address = g(row, "address", "street_address", "street address", "addr", "full_address", "location", default="")
        city    = g(row, "city", "town", "locality", default="")
        state   = g(row, "state", "state_code", "region", "province", default="")
        location_line = f"{address}, {city}, {state}".strip(", ").replace(" ,", ",")

        # Meta
        bedrooms  = g(row, "Bedrooms", "bedrooms", default="")
        bathrooms = g(row, "Bathrooms", "bathrooms", default="")
        sqft      = g(row, "Square Feet", "square_feet", "sqft", default="")
        sqn = _coerce_number(sqft)
        sqft_disp = f"{int(sqn):,}" if (sqn is not None and not math.isnan(sqn)) else (str(sqft) if sqft else "")

        meta_line = f"🛏 {bedrooms} bed"
        if bathrooms:
            meta_line += f"{EM_SPACE*3}🛁 {bathrooms} bath"
        if sqft_disp:
            meta_line += f"{EM_SPACE*3}📐 {sqft_disp} sqft"

        # Price
        price_raw = g(row, "Price", "price", default="")
        price_num = _coerce_number(price_raw)
        price = f"${price_num:,.0f}" if price_num is not None else (price_raw or "N/A")

        # STRICT four-line card
        card = f"{idx}) {title}\n{location_line}\n{meta_line}\n💰 {price}\n"
        cards.append(card)

    return "\n".join(cards)

def try_beautify_properties(answer_text: str):
    """
    If the agent returned a markdown table that looks like property rows,
    format them as numbered cards (STRICT four-line format) without IDs.
    """
    rows = _parse_markdown_table(answer_text)
    if not rows:
        return None

    header_keys = {k.lower() for k in rows[0].keys()}
    if not (("title" in header_keys) and ("price" in header_keys)):
        return None

    for r in rows:
        for k in list(r.keys()):
            if k.strip().lower() in {"id", "uuid"}:
                r.pop(k, None)

    pretty = format_properties(rows)
    return pretty

def try_beautify_state_values(answer_text: str):
    """
    If the agent returned a simple two-column markdown table like:
      | State | Average Listing Price |
    convert it to clean lines:
      Arizona — 💰 $719,693
    Supports generic metric names (uses 💰 for price-ish columns, ⏱️ for days).
    """
    rows = _parse_markdown_table(answer_text)
    if not rows:
        return None

    # Must look like exactly two columns and include 'state' as first or second
    headers = list(rows[0].keys())
    if len(headers) != 2:
        return None

    h1, h2 = headers[0].lower(), headers[1].lower()
    if "state" not in (h1, h2):
        return None

    # Determine which is the metric column
    metric_header = headers[1] if h1 == "state" else headers[0]
    is_price = any(tok in metric_header.lower() for tok in ["price", "per square foot", "$"])
    is_days  = any(tok in metric_header.lower() for tok in ["day", "dom"])

    lines = []
    for r in rows[:10]:
        state = r.get("State") or r.get("state") or r.get(headers[0]) if headers[0].lower()=="state" else r.get(headers[1])
        value = r.get(metric_header)

        prefix = "—"
        if is_price:
            prefix = "— 💰"
        elif is_days:
            prefix = "— ⏱️"

        lines.append(f"{state} {prefix} {value}")

    return "\n".join(lines)

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

        # Try to beautify property results into STRICT card style (numbered, spaced)
        pretty = try_beautify_properties(answer)
        if pretty:
            answer = "Here are the top matching properties (showing up to 10):\n\n" + pretty
        else:
            # Try to beautify state/value aggregates (two-column tables)
            pretty_state = try_beautify_state_values(answer)
            if pretty_state:
                answer = "Here are the results:\n\n" + pretty_state

        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "SQL Agent API is running."}

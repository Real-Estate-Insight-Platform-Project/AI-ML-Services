import os
import re
from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional
from fastapi.middleware.cors import CORSMiddleware
from postgrest.exceptions import APIError

# Load environment variables
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key not found in .env file")

# --- Initialize Clients ---
genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # In production, change this to your frontend's domain
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---


class QueryRequest(BaseModel):
    query: str


class ClickedProperty(BaseModel):
    type: str = Field("click", literal=True)
    city: str
    price: int
    bedrooms: int
    bathrooms: Optional[float] = None


class SearchQuery(BaseModel):
    type: str = Field("search", literal=True)
    query: str


class RecommendRequest(BaseModel):
    history: List[Union[ClickedProperty, SearchQuery]]

# --- Helper Functions ---


def extract_sql_from_markdown(markdown_text: str) -> str:
    """Extracts and cleans SQL from a markdown block."""
    match = re.search(r'```sql\n(.*?)\n```', markdown_text, re.DOTALL)
    if match:
        return match.group(1).strip().rstrip(';')
    # Fallback for plain text, cleaning it up
    return markdown_text.strip().replace('`', '').replace(';', '')

# --- API Endpoints ---


@app.post("/query")
async def get_sql_query(request: QueryRequest):
    """Handles a direct, single AI search query."""
    sql_query = ""
    try:
        prompt = f"""
        Translate the user query into a SQL SELECT statement for the "properties" table.
        RULES:
        1. ONLY generate `SELECT` statements.
        2. Use ONLY columns from this schema: `id`, `title`, `price`, `city`, `state`, `property_type`, `bedrooms`, `bathrooms`, `square_feet`, `year_built`, `listing_status`.
        3. ALWAYS use lowercase values for `property_type` (e.g., 'house', 'condo').
        4. Wrap the final SQL in a markdown block.
        User Query: "{request.query}"
        """
        response = model.generate_content(prompt)
        sql_query = extract_sql_from_markdown(response.text)

        if not sql_query.upper().startswith("SELECT"):
            raise HTTPException(
                status_code=400, detail="Invalid query. Only SELECT statements are allowed.")

        result = supabase.rpc(
            'execute_sql', {'sql_query': sql_query}).execute()
        return {"sql_query": sql_query, "data": result.data or []}
    except APIError as e:
        error_payload = e.json()
        error_message = error_payload.get('message', 'Database query error.')
        raise HTTPException(status_code=400, detail=error_message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}")


@app.post("/recommend")
async def get_recommendations(request: RecommendRequest):
    """Generates and executes a SQL query based on weighted user interaction history."""
    if not request.history:
        # This case is now handled client-side, but keep as a fallback.
        return {"sql_query": "No history provided.", "data": []}

    sql_query = ""
    try:
        # Format history for the prompt
        history_items = []
        for item in request.history:
            if item.type == 'search':
                history_items.append(f"a search for '{item.query}'")
            elif item.type == 'click':
                history_items.append(
                    f"a click on a property in {item.city} with {item.bedrooms} beds, {item.bathrooms} baths, and a price around ${item.price}")

        recent_history = history_items[:3]  # Top 3 are most recent
        older_history = history_items[3:7]  # Next 4 are older

        prompt_parts = [
            "Generate a single SQL SELECT query to recommend properties from the 'properties' table based on a user's browsing history.",
            "The query MUST include all properties, but rank them based on relevance.",
        ]

        if recent_history:
            prompt_parts.append(
                f"The user's MOST RECENT interests are: {'; '.join(recent_history)}. These are the most important criteria.")
        if older_history:
            prompt_parts.append(
                f"Their PAST interests include: {'; '.join(older_history)}. These are less important.")

        prompt_parts.append("""
        Construct a `CASE` statement to create a `ranking` column:
        - When a property matches any of the MOST RECENT interests, give it a `ranking` of 3.
        - When a property matches any of the PAST interests (but not recent ones), give it a `ranking` of 2.
        - All other properties get a `ranking` of 1.

        The final query must select all original columns plus the new `ranking` column.
        It MUST be ordered by `ranking` DESC, then by `created_at` DESC to show newer properties first.
        Limit the final result to 25 properties.

        Example of a CASE statement structure for a single condition:
        `CASE WHEN (city ILIKE '%Austin%' AND bedrooms >= 3) THEN 3 ... ELSE 1 END as ranking`

        **Final Rules:**
        1. The query MUST return properties that do not match any criteria (those with `ranking` = 1).
        2. Only generate a `SELECT` statement.
        3. Always use lowercase for `property_type`. For city comparisons, use `ILIKE` for case-insensitivity.
        4. Wrap the final SQL in a markdown block.
        """)

        prompt = "\n".join(prompt_parts)

        response = model.generate_content(prompt)
        sql_query = extract_sql_from_markdown(response.text)

        if not sql_query.upper().startswith("SELECT"):
            raise HTTPException(
                status_code=400, detail="Invalid recommendation query generated.")

        result = supabase.rpc(
            'execute_sql', {'sql_query': sql_query}).execute()
        return {"sql_query": sql_query, "data": result.data or []}
    except APIError as e:
        error_payload = e.json()
        error_message = error_payload.get(
            'message', 'DB recommendation query error.')
        raise HTTPException(status_code=400, detail=error_message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An internal recommendation error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)

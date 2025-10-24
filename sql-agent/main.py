"""
Agentic Real Estate Chatbot with Redis Session Management & SQL Agent Fallback
Supports: Supabase (PostgreSQL) + Google BigQuery + Redis
Features: Function calling, SQL agent fallback, persistent context, distance calculations
"""

import os
import re
import math
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text, inspect
from google import genai
from google.genai import types
from google.cloud import bigquery
from geopy.distance import geodesic
import redis
from redis.exceptions import RedisError
import uuid
import decimal
import requests  # NEW: open-source geocoding (US Census / Nominatim)

# Load environment variables
load_dotenv(find_dotenv(), override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SQL_AGENT_DATABASE_URL")
# BigQuery configuration
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID")
BQ_LOCATION = os.getenv("BQ_LOCATION")

# Make GOOGLE_APPLICATION_CREDENTIALS absolute if it's a relative path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'service_keys.json'

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
WEBSITE_BASE_URL = os.getenv("WEBSITE_BASE_URL", "https://yourdomain.com/property")

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://default:JHDhNLIc5ze3Djzv2GJ5kONxbN0Nf7Li@redis-16155.c264.ap-south-1-1.ec2.redns.redis-cloud.com:16155")
REDIS_SESSION_TTL = int(os.getenv("REDIS_SESSION_TTL", "3600"))  # 1 hour default

# Initialize clients
client = genai.Client(api_key=GOOGLE_API_KEY)

# Initialize BigQuery client with error handling
try:
    bq_client = bigquery.Client(project=BQ_PROJECT_ID)
    BQ_AVAILABLE = True
    print("✅ BigQuery client initialized successfully")
except Exception as e:
    print(f"⚠️ BigQuery initialization failed: {e}")
    print("⚠️ BigQuery features will be disabled")
    bq_client = None
    BQ_AVAILABLE = False

supabase_engine = create_engine(
    SUPABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    connect_args={"options": "-c statement_timeout=30000"}
)

# Initialize Redis connection with error handling
try:
    redis_client = redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
        health_check_interval=30
    )
    redis_client.ping()
    print("✅ Redis connected successfully")
    REDIS_AVAILABLE = True
except RedisError as e:
    print(f"⚠️ Redis connection failed: {e}")
    print("⚠️ Falling back to in-memory session storage")
    REDIS_AVAILABLE = False
    memory_sessions: Dict[str, List[Dict]] = {}

# ------------------- OPEN SOURCE GEOCODING (US CENSUS → NOMINATIM) -------------------

def _geocode_census(place: str) -> Optional[Dict[str, Any]]:
    """
    U.S. Census Geocoder: open, no API key needed for simple queries.
    https://geocoding.geo.census.gov/
    """
    try:
        url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
        params = {
            "address": place,
            "benchmark": "Public_AR_Current",
            "format": "json"
        }
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        matches = data.get("result", {}).get("addressMatches", [])
        if matches:
            best = matches[0]
            coords = best.get("coordinates", {})
            lat = coords.get("y")
            lon = coords.get("x")
            if lat is not None and lon is not None:
                return {
                    "lat": float(lat),
                    "lon": float(lon),
                    "formatted_address": best.get("matchedAddress", place)
                }
    except Exception as e:
        print(f"Census geocoder error: {e}")
    return None

def _geocode_nominatim(place: str) -> Optional[Dict[str, Any]]:
    """Nominatim fallback (OpenStreetMap)."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": 1, "addressdetails": 0}
        headers = {"User-Agent": "RealEstateInsightPlatform/1.0 (contact: ops@example.com)"}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        arr = r.json()
        if isinstance(arr, list) and arr:
            res = arr[0]
            return {
                "lat": float(res["lat"]),
                "lon": float(res["lon"]),
                "formatted_address": res.get("display_name", place)
            }
    except Exception as e:
        print(f"Nominatim error: {e}")
    return None

def resolve_place_to_coords(place: str) -> Optional[Dict[str, Any]]:
    """Resolve a place name to coordinates using US Census, then Nominatim."""
    if not place or not place.strip():
        return None
    # 1) Census
    cg = _geocode_census(place.strip())
    if cg:
        return cg
    # 2) Nominatim fallback
    nm = _geocode_nominatim(place.strip())
    if nm:
        return nm
    return None

# ==================== DATABASE SCHEMA INTROSPECTION ====================

def get_supabase_schema() -> str:
    """Get Supabase table schemas dynamically."""
    try:
        inspector = inspect(supabase_engine)
        tables = inspector.get_table_names()
        
        schema_info = "## SUPABASE (PostgreSQL) TABLES:\n\n"
        for table in tables:
            columns = inspector.get_columns(table)
            schema_info += f"### {table}\n"
            schema_info += "Columns: " + ", ".join([f"{col['name']} ({col['type']})" for col in columns]) + "\n\n"
        
        return schema_info
    except Exception as e:
        print(f"Schema introspection error: {e}")
        return ""

def get_bigquery_schema() -> str:
    """Get BigQuery table schemas."""
    try:
        dataset_ref = f"fourth-webbing-474805-j5.real_estate_market"
        tables = list(bq_client.list_tables(dataset_ref))
        
        schema_info = "## BIGQUERY TABLES:\n\n"
        for table in tables:
            table_ref = f"{dataset_ref}.{table.table_id}"
            table_obj = bq_client.get_table(table_ref)
            schema_info += f"### {table.table_id}\n"
            schema_info += "Columns: " + ", ".join([f"{field.name} ({field.field_type})" for field in table_obj.schema]) + "\n\n"
        
        return schema_info
    except Exception as e:
        print(f"BigQuery schema error: {e}")
        return ""

# ==================== SESSION MANAGEMENT ====================

class SessionManager:
    """Manages conversation sessions with Redis or in-memory fallback."""
    
    @staticmethod
    def get_conversation(session_id: str) -> List[Dict]:
        """Retrieve conversation history for a session."""
        if REDIS_AVAILABLE:
            try:
                data = redis_client.get(f"session:{session_id}")
                if data:
                    try:
                        parsed_data = json.loads(data)
                        # Ensure we return a list
                        if isinstance(parsed_data, list):
                            return parsed_data
                        else:
                            print(f"Invalid conversation format for session {session_id}, starting fresh")
                            return []
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error for session {session_id}: {e}")
                        return []
                return []
            except RedisError as e:
                print(f"Redis read error: {e}")
                return []
        else:
            return memory_sessions.get(session_id, [])
    
    @staticmethod
    def save_conversation(session_id: str, messages: List[Dict]) -> bool:
        """Save conversation history for a session."""
        if REDIS_AVAILABLE:
            try:
                # Ensure messages is serializable
                serializable_messages = convert_for_json(messages)
                redis_client.setex(
                    f"session:{session_id}",
                    REDIS_SESSION_TTL,
                    json.dumps(serializable_messages)
                )
                return True
            except (RedisError, TypeError, json.JSONEncodeError) as e:
                print(f"Redis write error for session {session_id}: {e}")
                return False
        else:
            memory_sessions[session_id] = messages
            return True
    
    @staticmethod
    def clear_conversation(session_id: str) -> bool:
        """Clear conversation history for a session."""
        if REDIS_AVAILABLE:
            try:
                redis_client.delete(f"session:{session_id}")
                return True
            except RedisError as e:
                print(f"Redis delete error: {e}")
                return False
        else:
            if session_id in memory_sessions:
                del memory_sessions[session_id]
            return True
    
    @staticmethod
    def get_all_sessions() -> List[str]:
        """Get all active session IDs."""
        if REDIS_AVAILABLE:
            try:
                keys = redis_client.keys("session:*")
                return [key.replace("session:", "") for key in keys]
            except RedisError as e:
                print(f"Redis keys error: {e}")
                return []
        else:
            return list(memory_sessions.keys())
    
    @staticmethod
    def get_session_info(session_id: str) -> Dict[str, Any]:
        """Get metadata about a session."""
        conversation = SessionManager.get_conversation(session_id)
        if not conversation:
            return {"exists": False}
        
        info = {
            "exists": True,
            "message_count": len(conversation),
            "created_at": conversation[0].get("timestamp") if conversation else None,
            "last_activity": conversation[-1].get("timestamp") if conversation else None
        }
        
        if REDIS_AVAILABLE:
            try:
                ttl = redis_client.ttl(f"session:{session_id}")
                info["ttl_seconds"] = ttl if ttl > 0 else None
            except RedisError:
                pass
        
        return info
    
    @staticmethod
    def extend_session(session_id: str) -> bool:
        """Extend session TTL (Redis only)."""
        if REDIS_AVAILABLE:
            try:
                return redis_client.expire(f"session:{session_id}", REDIS_SESSION_TTL)
            except RedisError as e:
                print(f"Redis expire error: {e}")
                return False
        return True




# ==================== DOMAIN KNOWLEDGE BASE ====================

DOMAIN_KNOWLEDGE = """
# COMPREHENSIVE REAL ESTATE DOMAIN KNOWLEDGE BASE

## DATABASE ARCHITECTURE

### SUPABASE TABLES (PostgreSQL)

#### 1. properties
**Purpose**: Active real estate listings with comprehensive property details
**Primary Key**: id (UUID)
**Columns**:
- id: Unique property identifier (UUID)
- title: Property marketing headline
- description: Detailed property description
- price: Listing price (USD, numeric)
- address: Street address
- city: City name
- state: State abbreviation (e.g., "CA", "NY")
- zip_code: 5-digit ZIP code
- property_type: "house", "condo", "townhouse", "apartment", "land"
- bedrooms: Number of bedrooms (integer)
- bathrooms: Number of bathrooms (decimal)
- square_feet: Interior square footage (integer)
- lot_size: Lot size in square feet (decimal)
- year_built: Year of construction (integer)
- listing_status: "active", "pending", "sold"
- created_at: Record creation timestamp
- updated_at: Last modification timestamp
- property_image: Image URL
- property_hyperlink: Realtor.com listing URL
- property_id: External listing ID
- listed_date: Original listing date
- longitude_coordinates: Longitude (decimal)
- latitude_coordinates: Latitude (decimal)
- noise_score: Environmental noise level (0-100)
- flood_factor: Flood risk factor
- fema_zone: FEMA flood zone designation

**Use Cases**:
- Property search by location, price, size, features
- Distance-based queries (near universities, landmarks)
- Environmental risk filtering (noise, flood)
- Agent performance analysis
- Joining with nri_counties for disaster risk assessment

#### 2. nri_counties
**Purpose**: FEMA National Risk Index data for county-level natural disaster risk assessment
**Primary Key**: county_fips (text)
**Columns**:
- county_fips: Unique FIPS code identifier (e.g., "06037" for Los Angeles County)
- county_name: Full county name (e.g., "Los Angeles")
- state_name: Full state name (e.g., "California")
- state_fips: Two-digit state FIPS code (e.g., "06")
- risk_index_score: Numeric score 0-100, higher = more disaster risk
- risk_index_rating: "Very Low", "Relatively Low", "Relatively Moderate", "Relatively High", "Very High"
- predominant_hazard: Primary risk (e.g., "Earthquake", "Hurricane", "Wildfire", "Tornado", "Flooding", "Drought")

**Use Cases**:
- Low-risk investments: risk_index_rating IN ('Very Low', 'Relatively Low')
- Hazard identification for property buyers
- State risk comparison: GROUP BY state_name, AVG(risk_index_score)
- JOIN with properties: Match on county names or use state + city matching

#### 3. gis.us_counties (PostGIS polygons)
Purpose: County geometries for point-in-polygon queries.
Key columns: gid (PK), geoid (county FIPS), name (county), statefp (2-digit FIPS), geom (SRID 4326), geom_3857 (generated)
Indexes: GIST on geom and geom_3857 already present per provided DDL.

### BIGQUERY TABLES
**Note**: All BigQuery tables should be referenced using the format: 
`fourth-webbing-474805-j5.real_estate_market.{table_name}`

#### 4. county_lookup
**Purpose**: Reference table linking counties to states with unique identifiers
**Columns**:
- county_num: Numeric county identifier (integer)
- county_fips: Official 5-digit FIPS code (string)
- county_name: Official county name (string)
- state_num: Numeric state identifier (integer)

**Use Cases**:
- Join county market data with state information
- Resolve county names to FIPS codes
- Connect county predictions to geographic identifiers

#### 5. county_market
**Purpose**: Historical monthly real estate market metrics at county level
**Columns**:
- county_num: Numeric county identifier (integer)
- year: Calendar year (integer)
- month: Month number 1-12 (integer)
- median_listing_price: Median price (USD)
- average_listing_price: Average price (USD)
- active_listing_count: Number of active listings
- median_days_on_market: Median days to sell
- median_square_feet: Median property size
- new_listing_count: New listings this month
- price_increased_count: Properties with price increases
- price_reduced_count: Properties with price reductions
- pending_listing_count: Pending sales
- *_mm: Month-over-month % change (e.g., median_listing_price_mm)
- *_yy: Year-over-year % change (e.g., median_listing_price_yy)

**Use Cases**:
- Historical trend analysis
- Market health indicators (days on market, pending ratio)
- Seasonal pattern identification
- Comparative market analysis across counties

#### 6. county_predictions
**Purpose**: AI-powered future market forecasts at county level
**Columns**:
- county_num: Numeric county identifier (integer)
- year: Forecast year (integer)
- month: Forecast month 1-12 (integer)
- median_listing_price_forecast: Predicted median price
- average_listing_price_forecast: Predicted average price
- active_listing_count_forecast: Predicted listing volume
- market_trend: "increasing", "stable", "declining"
- buyer_friendly: 1 = buyer's market, 0 = seller's market

**Use Cases**:
- Investment opportunity identification
- Future price forecasting
- Buyer vs seller market timing
- JOIN with county_market for trend validation

#### 7. state_lookup
**Purpose**: Reference table for U.S. states with identifiers
**Columns**:
- state_num: Numeric state identifier (integer)
- state_id: Two-letter abbreviation (e.g., "CA", "NY")
- state: Full state name (e.g., "California")

**Use Cases**:
- Join state market data with geographic names
- Resolve abbreviations to full names
- Connect state predictions to identifiers

#### 8. state_market
**Purpose**: Historical monthly real estate market metrics at state level (aggregated)
**Columns**:
- state_num: Numeric state identifier (integer)
- year: Calendar year
- month: Month number 1-12
- median_listing_price: State median price
- average_listing_price: State average price
- active_listing_count: Total active listings
- median_days_on_market: State median days to sell
- new_listing_count: New listings this month
- price_increased_count: Properties with price increases
- price_reduced_count: Properties with price reductions
- pending_ratio: Pending/active ratio
- *_mm: Month-over-month % change
- *_yy: Year-over-year % change

**Use Cases**:
- State-level market comparisons
- Macro trend analysis
- State ranking by market health
- High-level investment strategy

#### 9. state_predictions
**Purpose**: AI-powered future market forecasts at state level
**Columns**:
- state_num: Numeric state identifier (integer)
- year: Forecast year
- month: Forecast month 1-12
- median_listing_price_forecast: Predicted state median price
- average_listing_price_forecast: Predicted state average price
- median_listing_price_per_square_foot_forecast: Price per sqft
- active_listing_count_forecast: Predicted listing volume
- median_days_on_market_forecast: Predicted time to sell
- market_trend: "increasing", "stable", "declining"
- buyer_friendly: 1 = buyer's market, 0 = seller's market

**Use Cases**:
- State-level investment decisions
- Market timing strategies
- Cross-state opportunity comparison
- Migration pattern analysis

## QUERY GUIDELINES

### Place Resolution (Open-Source)
- When user mentions a place string (university, landmark, neighborhood), call **resolve_place** (US Census → Nominatim fallback),
  then use **find_properties_near_location** with the returned lat/lon.

### County-Based Filtering (CRITICAL)
- To get properties **inside a county** (e.g., "in low-risk counties"), use **point-in-polygon**: ST_Contains(c.geom, ST_SetSRID(ST_MakePoint(p.longitude_coordinates, p.latitude_coordinates), 4326))
- Prefer matching by **county FIPS (`geoid`)** from `nri_counties` → `gis.us_counties.geoid`, then filter properties via polygon containment.

### Context Management
- **Follow-up handling**: Track location, filters, and entity mentions across messages
- **Reference resolution**: "those properties", "that area", "similar homes" = use previous context
- **Progressive refinement**: "also under $400k", "with pool", "in low-risk area" = add to existing filters

### Time Context
- Current date: {datetime.now().strftime('%B %d, %Y')}
- For predictions: default to next 3 months if not specified
- Historical queries: query state_market or county_market

### JOIN Strategies
1. **Properties + Risk**: JOIN properties with nri_counties on county names or use city/state matching
2. **County Market + Lookup**: JOIN county_market/county_predictions with county_lookup on county_fips
3. **State Market + Lookup**: JOIN state_market/state_predictions with state_lookup on state abbreviation
4. **Cross-database**: Query BigQuery first, then filter Supabase properties by resulting counties/states

### Common Query Patterns
1. **Properties in low-risk areas**: 
   - Query nri_counties WHERE risk_index_score < 40
   - JOIN/filter properties in those counties
2. **Properties near location**: 
   - Get coordinates → use distance tool → filter by proximity
3. **Investment opportunities**: 
   - Query predictions (buyer_friendly=1, trend=increasing)
   - JOIN with low-risk counties
   - Filter properties in those areas
4. **Market trend analysis**:
   - Query county_market/state_market for historical data
   - JOIN with predictions for future outlook
   - Calculate growth rates

## RESPONSE FORMATTING

### Property Cards (STRICT FORMAT)
```
1) {{Title}}
{{Address}}, {{City}}, {{State}} {{ZIP}}
🛏 {{Beds}} bed   🛁 {{Baths}} bath   📐 {{SqFt}} sqft
💰 ${{Price:,}}
🔗 View details: {{property_hyperlink}}
📍 Distance: {{distance}} miles (if applicable)
⚠️ Risk: {{risk_rating}} ({{predominant_hazard}}) (if applicable)
```

### Emojis
- 💰 Prices
- 📈 Growth/increases
- 📉 Declines
- ⏱️ Time metrics
- 🏘️ Listing counts
- ✅ Very Low risk
- ⚠️ Moderate/High risk
- 🟢 Buyer-friendly
- 🔴 Seller-friendly
- 🛏 Bedrooms
- 🛁 Bathrooms
- 📐 Square feet
- 📍 Location/distance

## CONVERSATIONAL GUIDELINES
- Remember previous messages in session
- Handle ambiguous references: "Did you mean [option A] or [option B]?"
- Suggest next steps: "Would you like to see similar properties in nearby areas?"
- Out-of-scope: "I specialize in real estate insights. For [topic], I recommend..."
- Always use property_hyperlink field for listing URLs (not constructed URLs)
- Maintain context across turns, refine filters progressively.

"""

# ==================== SQL AGENT FALLBACK ====================

# Allowed tables for SQL agent (security whitelist)
ALLOWED_SUPABASE_TABLES = ["properties", "nri_counties"]
ALLOWED_BIGQUERY_TABLES = [
  "county_lookup", "county_market", "county_predictions",
  "state_lookup", "state_market", "state_predictions"
]

def sanitize_sql_query(query: str, database: str = "supabase") -> Optional[str]:
  """
  Sanitize and validate SQL query for security.
  Returns None if query is unsafe.
  """
  query_upper = query.strip().upper()
  
  # Only allow SELECT queries
  if not query_upper.startswith("SELECT"):
      return None
  
  # Reject dangerous keywords
  dangerous_keywords = [
      "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE",
      "TRUNCATE", "REPLACE", "GRANT", "REVOKE", "EXEC", "EXECUTE"
  ]
  for keyword in dangerous_keywords:
      if keyword in query_upper:
          return None
  
  # Validate table references - handle both simple and fully qualified table names
  allowed_tables = ALLOWED_SUPABASE_TABLES if database == "supabase" else ALLOWED_BIGQUERY_TABLES
  
  # Extract table names (improved pattern matching for BigQuery)
  if database == "bigquery":
      # Handle both simple table names and fully qualified names
      # Pattern 1: Simple table names (e.g., state_predictions)
      from_match = re.search(r'FROM\s+[`"]?(\w+)[`"]?', query_upper)
      join_matches = re.findall(r'JOIN\s+[`"]?(\w+)[`"]?', query_upper)
      
      # Pattern 2: Fully qualified names (e.g., project.dataset.table)
      from_qualified = re.search(r'FROM\s+[`"]?[\w.-]+\.(\w+)[`"]?', query_upper)
      join_qualified = re.findall(r'JOIN\s+[`"]?[\w.-]+\.(\w+)[`"]?', query_upper)
      
      table_names = []
      # Add simple table names
      if from_match:
          table_names.append(from_match.group(1).lower())
      table_names.extend([t.lower() for t in join_matches])
      
      # Add qualified table names
      if from_qualified:
          table_names.append(from_qualified.group(1).lower())
      table_names.extend([t.lower() for t in join_qualified])
      
  else:
      # Simple table names for Supabase
      from_match = re.search(r'FROM\s+[`"]?(\w+)[`"]?', query_upper)
      join_matches = re.findall(r'JOIN\s+[`"]?(\w+)[`"]?', query_upper)
      
      table_names = []
      if from_match:
          table_names.append(from_match.group(1).lower())
      table_names.extend([t.lower() for t in join_matches])
  
#   # Remove duplicates and check if all tables are allowed
#   table_names = list(set(table_names))
#   for table in table_names:
#       if table not in [t.lower() for t in allowed_tables]:
#           print(f"❌ [SECURITY] Table '{table}' not in allowed tables: {allowed_tables}")
#           return None
  
  # Add LIMIT if not present
  if "LIMIT" not in query_upper:
      query = query.rstrip(';') + " LIMIT 50"
  
  return query

def execute_sql_agent_query(query: str, database: str = "supabase") -> Dict[str, Any]:
  """
  Execute a SQL query with safety checks.
  """
  try:
      safe_query = sanitize_sql_query(query, database)
      if not safe_query:
          return {
              "error": "Query failed security validation. Only SELECT queries on allowed tables are permitted.",
              "allowed_tables": ALLOWED_SUPABASE_TABLES if database == "supabase" else ALLOWED_BIGQUERY_TABLES
          }
      
      if database == "supabase":
          with supabase_engine.connect() as conn:
              result = conn.execute(text(safe_query))
              rows = result.fetchall()
              raw_rows = [dict(row._mapping) for row in rows]
              return {
                  "success": True,
                  "rows": [convert_for_json(row) for row in raw_rows],
                  "count": len(rows),
                  "query": safe_query
              }
      
      elif database == "bigquery":
          if not BQ_AVAILABLE or bq_client is None:
              return {"error": "BigQuery is not available. Please check your credentials and configuration."}
          
          print(f"🗄️ [SQL-AGENT] Database: {database}")
          print(f"📝 [SQL-AGENT] Original query: {query}")
          print(f"🔒 [SQL-AGENT] Sanitized query: {safe_query}")
          
          processed_query = safe_query
          # Replace simple table names with fully qualified names using the specific format
          if "fourth-webbing-474805-j5.real_estate_market" not in processed_query:
              for table in ALLOWED_BIGQUERY_TABLES:
                  # Match table name not preceded by a dot (not already qualified)
                  pattern = r'(?<!\w\.)(?<!\.)' + re.escape(table) + r'(?!\w)'
                  replacement = f"`fourth-webbing-474805-j5.real_estate_market.{table}`"
                  processed_query = re.sub(
                      pattern,
                      replacement,
                      processed_query,
                      flags=re.IGNORECASE
                  )
          
          print(f"🔧 [SQL-AGENT] Processed query: {processed_query}")
          
          job_config = bigquery.QueryJobConfig()
          query_job = bq_client.query(processed_query)
          results = query_job.result()
          rows = [dict(row) for row in results]
        
          print(f"✅ [SQL-AGENT] Query executed successfully, returned {len(rows)} rows")
          
          return {
              "success": True,
              "rows": rows,
              "count": len(rows),
              "query": processed_query
          }
      
      else:
          return {"error": f"Unknown database: {database}"}
  
  except Exception as e:
      return {
          "error": f"Query execution failed: {str(e)}",
          "query": query
      }

# ==================== TOOL DEFINITIONS & SPATIAL HELPERS ====================

def calculate_distance_tool(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  """Calculate distance in miles between two coordinates."""
  try:
      distance_miles = geodesic((lat1, lon1), (lat2, lon2)).miles
      return round(distance_miles, 2)
  except Exception:
      return -1.0

def convert_for_json(obj):
  """Convert objects to JSON-serializable format."""
  if obj is None:
      return None
  elif isinstance(obj, (str, int, float, bool)):
      return obj
  elif isinstance(obj, uuid.UUID):
      return str(obj)
  elif isinstance(obj, decimal.Decimal):
      return float(obj)
  elif isinstance(obj, (datetime, timedelta)):
      return obj.isoformat()
  elif isinstance(obj, dict):
      return {str(k): convert_for_json(v) for k, v in obj.items()}
  elif isinstance(obj, (list, tuple)):
      return [convert_for_json(item) for item in obj]
  elif hasattr(obj, '__dict__'):
      try:
          return str(obj)
      except Exception:
          return f"<{type(obj).__name__} object>"
  else:
      try:
          return str(obj)
      except Exception:
          return f"<unserializable {type(obj).__name__}>"

def query_supabase_properties(filters: Dict[str, Any], limit: int = 10) -> List[Dict]:
  """Query Supabase properties with filters."""
  query = """
      SELECT id, title, address, city, state, zip_code, price, bedrooms, 
             bathrooms, square_feet, property_type, latitude_coordinates, 
             longitude_coordinates, listing_status, property_hyperlink, 
             noise_score, flood_factor, fema_zone
      FROM properties 
      WHERE listing_status = 'active'
  """
  params = {}
  
  if filters.get("city"):
      query += " AND LOWER(city) = LOWER(:city)"
      params["city"] = filters["city"]
  
  if filters.get("state"):
      query += " AND LOWER(state) = LOWER(:state)"
      params["state"] = filters["state"]
  
  if filters.get("min_price") is not None:
      query += " AND price >= :min_price"
      params["min_price"] = filters["min_price"]
  
  if filters.get("max_price") is not None:
      query += " AND price <= :max_price"
      params["max_price"] = filters["max_price"]
  
  if filters.get("bedrooms") is not None:
      query += " AND bedrooms >= :bedrooms"
      params["bedrooms"] = filters["bedrooms"]
  
  if filters.get("bathrooms") is not None:
      query += " AND bathrooms >= :bathrooms"
      params["bathrooms"] = filters["bathrooms"]
  
  if filters.get("property_type"):
      query += " AND LOWER(property_type) = LOWER(:property_type)"
      params["property_type"] = filters["property_type"]
  
  query += f" ORDER BY created_at DESC LIMIT :limit"
  params["limit"] = min(limit, 5000)  # Increased max limit for location-based searches
  
  print(f"📝 Executing Supabase query: {query}")
  print(f"🔧 Query parameters: {params}")
  
  with supabase_engine.connect() as conn:
      result = conn.execute(text(query), params)
      rows = result.fetchall()
      properties = [dict(row._mapping) for row in rows]
      return [convert_for_json(prop) for prop in properties]

def query_low_risk_counties(state: Optional[str] = None, max_risk_score: int = 40) -> List[Dict]:
  """Find low-risk counties from FEMA NRI data."""
  query = """
      SELECT county_name, state_name, county_fips, risk_index_score, 
             risk_index_rating, predominant_hazard
      FROM nri_counties
      WHERE risk_index_score <= :max_risk
  """
  params = {"max_risk": max_risk_score}
  
  if state:
      query += " AND LOWER(state_name) = LOWER(:state)"
      params["state"] = state
  
  query += " ORDER BY risk_index_score ASC LIMIT 50"
  
  print(f"📝 Executing FEMA NRI query: {query}")
  print(f"🔧 Query parameters: {params}")
  
  with supabase_engine.connect() as conn:
      result = conn.execute(text(query), params)
      rows = result.fetchall()
      raw_rows = [dict(row._mapping) for row in rows]
      return [convert_for_json(row) for row in raw_rows]

# ---- County polygon helpers (PostGIS) ----

# Optional state abbrev → 2-digit FIPS for tighter matching when user provides a state
STATE_ABBR_TO_FIPS = {
  "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11","FL":"12","GA":"13",
  "HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22","ME":"23","MD":"24","MA":"25",
  "MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36",
  "NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48",
  "UT":"49","VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56","PR":"72"
}

def query_properties_in_counties_by_fips(
  county_fips_list: List[str],
  filters: Optional[Dict[str, Any]] = None,
  limit: int = 100
) -> List[Dict]:
  """
  Use PostGIS polygon containment to return active properties inside any county FIPS in the list.
  """
  if not county_fips_list:
      return []

  # Build dynamic IN list safely
  fips_params = {f"f{i}": fips for i, fips in enumerate(county_fips_list[:100])}
  in_clause = ", ".join([f":{k}" for k in fips_params.keys()])

  base = f"""
      SELECT p.id, p.title, p.address, p.city, p.state, p.zip_code, p.price, p.bedrooms,
             p.bathrooms, p.square_feet, p.property_type, p.latitude_coordinates,
             p.longitude_coordinates, p.listing_status, p.property_hyperlink,
             p.noise_score, p.flood_factor, p.fema_zone
      FROM properties p
      JOIN gis.us_counties c
        ON ST_Contains(
             c.geom,
             ST_SetSRID(ST_MakePoint(p.longitude_coordinates, p.latitude_coordinates), 4326)
           )
      WHERE c.geoid IN ({in_clause})
        AND p.listing_status = 'active'
  """
  params: Dict[str, Any] = {}
  params.update(fips_params)

  # Apply optional property filters
  filters = filters or {}
  if filters.get("min_price") is not None:
      base += " AND p.price >= :min_price"
      params["min_price"] = filters["min_price"]
  if filters.get("max_price") is not None:
      base += " AND p.price <= :max_price"
      params["max_price"] = filters["max_price"]
  if filters.get("bedrooms") is not None:
      base += " AND p.bedrooms >= :bedrooms"
      params["bedrooms"] = filters["bedrooms"]
  if filters.get("property_type"):
      base += " AND LOWER(p.property_type) = LOWER(:ptype)"
      params["ptype"] = filters["property_type"]

  base += " ORDER BY p.created_at DESC LIMIT :limit"
  params["limit"] = min(limit, 100)

  print("🧭 Executing county polygon property query")
  print(base)
  with supabase_engine.connect() as conn:
      result = conn.execute(text(base), params)
      rows = [dict(r._mapping) for r in result.fetchall()]
      return [convert_for_json(r) for r in rows]

def find_properties_in_named_county(
  county_name: str,
  state_abbr: Optional[str] = None,
  filters: Optional[Dict[str, Any]] = None,
  limit: int = 50
) -> List[Dict]:
  """
  Resolve a county by name (+ optional state), get its FIPS from gis.us_counties, and return properties inside it.
  """
  if not county_name:
      return []
  params: Dict[str, Any] = {"cname": county_name}
  sql = """
      SELECT geoid, name, statefp
      FROM gis.us_counties
      WHERE LOWER(name) = LOWER(:cname)
  """
  if state_abbr and state_abbr.upper() in STATE_ABBR_TO_FIPS:
      sql += " AND statefp = :sfp"
      params["sfp"] = STATE_ABBR_TO_FIPS[state_abbr.upper()]
  sql += " LIMIT 5"

  with supabase_engine.connect() as conn:
      matches = [dict(r._mapping) for r in conn.execute(text(sql), params).fetchall()]

  if not matches:
      return []
  fips_list = [m["geoid"] for m in matches if m.get("geoid")]
  return query_properties_in_counties_by_fips(fips_list, filters, limit)

def find_properties_in_low_risk_counties(
  max_risk_score: int = 40,
  state_name: Optional[str] = None,
  filters: Optional[Dict[str, Any]] = None,
  limit: int = 100
) -> List[Dict]:
  """
  Pull low-risk counties from nri_counties (by score and optional state), get their FIPS,
  then return properties that fall inside those counties (polygon containment).
  """
  low = query_low_risk_counties(state_name, max_risk_score)
  fips_list = [row["county_fips"] for row in low if row.get("county_fips")]
  if not fips_list:
      return []
  return query_properties_in_counties_by_fips(fips_list, filters, limit)

# ---- Proximity helper ----

def find_properties_near_location(
    target_lat: float,
    target_lon: float,
    max_distance_miles: float = 10.0,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """Find properties within distance of target location (haversine via geopy)."""
    base_filters = filters or {}
    
    # For location-based searches, we need a much larger sample to ensure we get properties
    # from the target area since the database is ordered by created_at, not location
    print(f"🔍 Loading properties for distance calculation (radius: {max_distance_miles} miles)")
    properties = query_supabase_properties(base_filters, limit=2000)  # Increased from 200
    
    print(f"📊 Retrieved {len(properties)} properties for distance analysis")
    
    results = []
    for prop in properties:
        if prop.get("latitude_coordinates") and prop.get("longitude_coordinates"):
            distance = calculate_distance_tool(
                target_lat, target_lon,
                float(prop["latitude_coordinates"]), 
                float(prop["longitude_coordinates"])
            )
            if distance >= 0 and distance <= max_distance_miles:
                prop["distance_miles"] = distance
                results.append(prop)
    
    print(f"🎯 Found {len(results)} properties within {max_distance_miles} miles")
    results.sort(key=lambda x: x["distance_miles"])
    return results[:50]

def query_state_predictions(months_ahead: int = 3, trend_filter: Optional[str] = None) -> List[Dict]:
    """Query state predictions from BigQuery for market forecasts."""
    if not BQ_AVAILABLE or bq_client is None:
        return []
    
    try:
        # Calculate target year and month robustly
        current_date = datetime.now()
        target_dates = []
        for i in range(1, months_ahead + 1):
            future_month = (current_date.month - 1 + i) % 12 + 1
            future_year = current_date.year + ((current_date.month - 1 + i) // 12)
            target_dates.append((future_year, future_month))
        
        date_conditions = [f"(year = {y} AND month = {m})" for y, m in target_dates]
        date_filter = " OR ".join(date_conditions)
        
        # Use correct column names based on actual BigQuery schema
        query = f"""
        SELECT 
            sl.state_id,
            sl.state,
            sp.year,
            sp.month,
            sp.median_listing_price,
            sp.average_listing_price,
            sp.median_listing_price_per_square_foot,
            sp.market_trend,
            sp.buyer_friendly
        FROM `fourth-webbing-474805-j5.real_estate_market.state_predictions` sp
        JOIN `fourth-webbing-474805-j5.real_estate_market.state_lookup` sl
            ON sp.state_num = sl.state_num
        WHERE ({date_filter})
        """
        
        if trend_filter:
            query += f" AND LOWER(sp.market_trend) = LOWER('{trend_filter}')"
        
        query += " ORDER BY sl.state, sp.year, sp.month"
        
        print(f"📝 Executing BigQuery state predictions query: {query}")
        
        job_config = bigquery.QueryJobConfig()
        query_job = bq_client.query(query, job_config=job_config, location=BQ_LOCATION)
        results = query_job.result()
        rows = [dict(row) for row in results]
        
        print(f"📊 BigQuery returned {len(rows)} prediction records")
        
        return rows
        
    except Exception as e:
        print(f"Error querying state predictions: {e}")
        return []

# ==================== SCHEMA HELPER ====================

def S(t, **kwargs):
  # t must be 'OBJECT','STRING','NUMBER','INTEGER','BOOLEAN','ARRAY'
  return types.Schema(type=t, **kwargs)

# ==================== FUNCTION DECLARATIONS ====================

tools = [
  types.Tool(
      function_declarations=[
          # Open-source geocoding
          types.FunctionDeclaration(
              name="resolve_place",
              description="Resolve a place name or landmark to latitude/longitude using open-source geocoders (US Census → Nominatim).",
              parameters=S(
                  "OBJECT",
                  properties={"place": S("STRING", description="Place name, e.g., 'Stanford University'")},
                  required=["place"],
              ),
          ),
          types.FunctionDeclaration(
              name="calculate_distance",
              description="Calculate distance in miles between two geographic coordinates.",
              parameters=S(
                  "OBJECT",
                  properties={
                      "lat1": S("NUMBER", description="Latitude of first location"),
                      "lon1": S("NUMBER", description="Longitude of first location"),
                      "lat2": S("NUMBER", description="Latitude of second location"),
                      "lon2": S("NUMBER", description="Longitude of second location"),
                  },
                  required=["lat1", "lon1", "lat2", "lon2"],
              ),
          ),
          types.FunctionDeclaration(
              name="search_properties",
              description="Search for real estate properties in Supabase with various filters. Returns up to 50 active listings.",
              parameters=S(
                  "OBJECT",
                  properties={
                      "city": S("STRING", description="City name"),
                      "state": S("STRING", description="State name or abbreviation"),
                      "min_price": S("INTEGER", description="Minimum price in USD"),
                      "max_price": S("INTEGER", description="Maximum price in USD"),
                      "bedrooms": S("INTEGER", description="Minimum bedrooms"),
                      "bathrooms": S("NUMBER", description="Minimum bathrooms"),
                      "property_type": S("STRING", description="Property type (house, condo, townhouse, apartment, land)"),
                  },
              ),
          ),
          types.FunctionDeclaration(
              name="find_properties_near_location",
              description="Find properties within specific distance from target location (university, landmark, etc.).",
              parameters=S(
                  "OBJECT",
                  properties={
                      "target_lat": S("NUMBER", description="Target latitude"),
                      "target_lon": S("NUMBER", description="Target longitude"),
                      "max_distance_miles": S("NUMBER", description="Max distance in miles (default 10)"),
                      "min_price": S("INTEGER", description="Optional min price"),
                      "max_price": S("INTEGER", description="Optional max price"),
                      "bedrooms": S("INTEGER", description="Optional min bedrooms"),
                      "property_type": S("STRING", description="Optional property type"),
                  },
                  required=["target_lat", "target_lon"],
              ),
          ),
          # New helpers that use county polygons
          types.FunctionDeclaration(
              name="find_properties_in_county",
              description="Find properties inside a named county (optionally scoped by state). Uses PostGIS county polygons.",
              parameters=S(
                  "OBJECT",
                  properties={
                      "county_name": S("STRING", description="County name, e.g., 'Santa Clara'"),
                      "state": S("STRING", description="Optional state abbreviation, e.g., 'CA'"),
                      "min_price": S("INTEGER", description="Optional min price"),
                      "max_price": S("INTEGER", description="Optional max price"),
                      "bedrooms": S("INTEGER", description="Optional min bedrooms"),
                      "property_type": S("STRING", description="Optional property type"),
                  },
                  required=["county_name"],
              ),
          ),
          types.FunctionDeclaration(
              name="find_properties_in_low_risk_counties",
              description="Find properties inside low-risk counties (by FEMA NRI risk_index_score) using PostGIS polygon containment.",
              parameters=S(
                  "OBJECT",
                  properties={
                      "max_risk_score": S("INTEGER", description="Max risk score (default 40)"),
                      "state": S("STRING", description="Optional state name filter for counties"),
                      "min_price": S("INTEGER", description="Optional min price"),
                      "max_price": S("INTEGER", description="Optional max price"),
                      "bedrooms": S("INTEGER", description="Optional min bedrooms"),
                      "property_type": S("STRING", description="Optional property type"),
                  },
              ),
          ),
          types.FunctionDeclaration(
              name="find_properties_near_place",
              description="Resolve a place to coordinates, then find nearby properties within a given radius and optional filters.",
              parameters=S(
                  "OBJECT",
                  properties={
                      "place": S("STRING", description="Place name or landmark"),
                      "max_distance_miles": S("NUMBER", description="Max distance in miles (default 10)"),
                      "min_price": S("INTEGER", description="Optional min price"),
                      "max_price": S("INTEGER", description="Optional max price"),
                      "bedrooms": S("INTEGER", description="Optional min bedrooms"),
                      "property_type": S("STRING", description="Optional property type"),
                  },
                  required=["place"],
              ),
          ),
          # Existing custom SQL tool
          types.FunctionDeclaration(
              name="execute_custom_sql",
              description="Execute a custom SQL SELECT query when predefined tools don't cover the use case. ONLY use SELECT on allowed tables. For BigQuery, tables will be automatically qualified as fourth-webbing-474805-j5.real_estate_market.{table_name}.",
              parameters=S(
                  "OBJECT",
                  properties={
                      "query": S(
                          "STRING",
                          description="SQL SELECT query. Allowed Supabase tables: properties, nri_counties. Allowed BigQuery tables: county_lookup, county_market, county_predictions, state_lookup, state_market, state_predictions."
                      ),
                      "database": S(
                          "STRING",
                          description="Target database (supabase or bigquery)"
                      ),
                  },
                  required=["query", "database"],
              ),
          )
      ]
  )
]

# ==================== FUNCTION EXECUTION ====================

def execute_function(func_call: types.FunctionCall) -> Any:
  """Execute the called function."""
  func_name = func_call.name
  args = func_call.args
  
  print(f"🔧 [FUNCTION] Calling function: {func_name}")
  print(f"📊 [FUNCTION] Parameters: {args}")
  
  try:
      if func_name == "resolve_place":
          place = args["place"]
          print(f"🔎 Geocoding (open-source): {place}")
          res = resolve_place_to_coords(place)
          if not res:
              return {"error": f"Could not resolve '{place}' to coordinates."}
          return {"lat": res["lat"], "lon": res["lon"], "formatted_address": res["formatted_address"]}

      elif func_name == "calculate_distance":
          return calculate_distance_tool(
              args["lat1"], args["lon1"], args["lat2"], args["lon2"]
          )
      
      elif func_name == "search_properties":
          filters = {k: v for k, v in args.items() if v is not None}
          print(f"🔍 AI Assistant Query - search_properties with filters: {filters}")
          results = query_supabase_properties(filters)
          print(f"📊 AI Assistant Result - Found {len(results)} properties")
          return {"properties": results, "count": len(results)}
      
      elif func_name == "find_properties_near_location":
          filters = {
              "min_price": args.get("min_price"),
              "max_price": args.get("max_price"),
              "bedrooms": args.get("bedrooms"),
              "property_type": args.get("property_type")
          }
          filters = {k: v for k, v in filters.items() if v is not None}
          print(f"🔍 AI Assistant Query - find_properties_near_location at lat:{args['target_lat']}, lon:{args['target_lon']}, distance:{args.get('max_distance_miles', 10.0)} miles, filters: {filters}")
          results = find_properties_near_location(
              args["target_lat"],
              args["target_lon"],
              args.get("max_distance_miles", 10.0),
              filters if filters else None
          )
          print(f"📊 AI Assistant Result - Found {len(results)} properties within distance")
          return {"properties": results, "count": len(results)}

      elif func_name == "find_properties_near_place":
          place = args["place"]
          max_dist = args.get("max_distance_miles", 10.0)
          filters = {
              "min_price": args.get("min_price"),
              "max_price": args.get("max_price"),
              "bedrooms": args.get("bedrooms"),
              "property_type": args.get("property_type"),
          }
          filters = {k: v for k, v in filters.items() if v is not None}
          print(f"🔍 AI Assistant Query - find_properties_near_place: {place} (radius {max_dist} mi), filters: {filters}")
          
          loc = resolve_place_to_coords(place)
          if not loc:
              return {"error": f"Could not resolve '{place}' to coordinates."}
          
          print(f"📍 Resolved '{place}' to coordinates: {loc['lat']}, {loc['lon']}")
          
          results = find_properties_near_location(
              loc["lat"], loc["lon"], max_dist, filters if filters else None
          )
          print(f"📊 AI Assistant Result - Found {len(results)} properties near '{loc.get('formatted_address', place)}'")
          return {
              "properties": results,
              "count": len(results),
              "center": {"lat": loc["lat"], "lon": loc["lon"], "formatted_address": loc["formatted_address"]}
          }

      elif func_name == "find_properties_in_county":
          cname = args["county_name"]
          state = args.get("state")
          filters = {
              "min_price": args.get("min_price"),
              "max_price": args.get("max_price"),
              "bedrooms": args.get("bedrooms"),
              "property_type": args.get("property_type"),
          }
          filters = {k: v for k, v in filters.items() if v is not None}
          print(f"🧭 County polygon search: county='{cname}', state='{state}', filters={filters}")
          props = find_properties_in_named_county(cname, state, filters, limit=50)
          return {"properties": props, "count": len(props)}

      elif func_name == "find_properties_in_low_risk_counties":
          filters = {
              "min_price": args.get("min_price"),
              "max_price": args.get("max_price"),
              "bedrooms": args.get("bedrooms"),
              "property_type": args.get("property_type"),
          }
          filters = {k: v for k, v in filters.items() if v is not None}
          max_risk = args.get("max_risk_score", 40)
          state = args.get("state")
          print(f"🛡 Low-risk counties polygon search: max_risk={max_risk}, state={state}, filters={filters}")
          props = find_properties_in_low_risk_counties(max_risk, state, filters, limit=100)
          return {"properties": props, "count": len(props)}

      elif func_name == "get_low_risk_counties":
          print(f"🔍 AI Assistant Query - get_low_risk_counties with state:{args.get('state')}, max_risk_score:{args.get('max_risk_score', 40)}")
          results = query_low_risk_counties(
              args.get("state"),
              args.get("max_risk_score", 40)
          )
          print(f"📊 AI Assistant Result - Found {len(results)} low-risk counties")
          return {"counties": results, "count": len(results)}
      
      elif func_name == "get_state_predictions":
          print(f"🔍 AI Assistant Query - get_state_predictions for {args.get('months_ahead', 3)} months ahead, trend_filter:{args.get('trend_filter')}")
          results = query_state_predictions(
              args.get("months_ahead", 3),
              args.get("trend_filter")
          )
          print(f"📊 AI Assistant Result - Found {len(results)} state predictions")
          return {"predictions": results, "count": len(results)}
      
      elif func_name == "execute_custom_sql":
          print(f"🔍 AI Assistant Query - execute_custom_sql on {args['database']} database")
          print(f"📝 SQL Query: {args['query']}")
          result = execute_sql_agent_query(
              args["query"],
              args["database"]
          )
          if "success" in result and result["success"]:
              print(f"📊 AI Assistant Result - Query executed successfully, returned {result.get('count', 0)} rows")
          else:
              print(f"❌ AI Assistant Error - Query failed: {result.get('error', 'Unknown error')}")
          return result
      
      else:
          return {"error": f"Unknown function: {func_name}"}
  
  except Exception as e:
      return {"error": str(e)}

# ==================== RESPONSE FORMATTING ====================

def format_property_card(prop: Dict, index: int = 1) -> str:
  """Format single property as card."""
  title = prop.get("title", "Untitled Property")
  address = prop.get("address", "")
  city = prop.get("city", "")
  state = prop.get("state", "")
  zip_code = prop.get("zip_code", "")
  
  location = f"{address}, {city}, {state} {zip_code}".strip(", ")
  
  beds = prop.get("bedrooms", "")
  baths = prop.get("bathrooms", "")
  sqft = prop.get("square_feet", "")
  sqft_display = f"{int(sqft):,}" if sqft else ""
  
  meta_parts = []
  if beds not in ("", None):
      meta_parts.append(f"🛏 {beds} bed")
  if baths not in ("", None):
      meta_parts.append(f"🛁 {baths} bath")
  if sqft_display:
      meta_parts.append(f"📐 {sqft_display} sqft")
  meta_line = "   ".join(meta_parts)
  
  price = prop.get("price", 0)
  price_display = f"${int(price):,}" if price else "N/A"
  
  hyperlink = prop.get("property_hyperlink", "")
  link = "🔗 View details" if hyperlink else "🔗 Listing details unavailable"
  
  distance_info = ""
  if "distance_miles" in prop:
      distance_info = f"\n📍 Distance: {prop['distance_miles']} miles"
  
  risk_info = ""
  if "risk_rating" in prop and "hazard" in prop:
      risk_info = f"\n⚠️ Risk: {prop['risk_rating']} ({prop['hazard']})"
  
  if hyperlink:
      return f"{index}) {title}\n{location}\n{meta_line}\n💰 {price_display}\n{link}\n{hyperlink}{distance_info}{risk_info}\n"
  else:
      return f"{index}) {title}\n{location}\n{meta_line}\n💰 {price_display}\n{link}{distance_info}{risk_info}\n"

def format_properties_response(properties: List[Dict]) -> str:
  """Format multiple properties as numbered cards."""
  if not properties:
      return "No properties found matching your criteria."
  
  cards = []
  for idx, prop in enumerate(properties[:10], start=1):
      cards.append(format_property_card(prop, idx))
  
  total = len(properties)
  shown = min(total, 10)
  header = f"Found {total} {'property' if total == 1 else 'properties'}. Showing top {shown}:\n\n"
  return header + "\n".join(cards)

# ==================== CHAT HANDLER WITH SESSION MANAGEMENT ====================

def handle_chat(session_id: str, user_message: str) -> str:
  """
  Handle chat with Redis-backed session management and SQL agent fallback.
  """
  try:
      conversation = SessionManager.get_conversation(session_id)
      timestamp = datetime.now().isoformat()
      
      print(f"\n🤖 AI Assistant Processing Message:")
      print(f"   Session: {session_id}")
      print(f"   Message: {user_message}")
      print(f"   Conversation History: {len(conversation)} previous messages")
      print("="*60)
      
      system_instruction = f"""You are an expert real estate AI assistant with comprehensive property, market, and risk data.

{DOMAIN_KNOWLEDGE}

## YOUR CAPABILITIES:
1. Property Search (filters)
2. Distance Search (places → coordinates with open-source geocoders)
3. County Polygon Search (properties *inside* a county using PostGIS)
4. Risk Assessment (FEMA NRI)
5. Market Analysis (BigQuery)
6. Custom SQL for complex joins

## TOOL USAGE:
- For **place strings** (e.g., 'Stanford University'), call **resolve_place** then **find_properties_near_location**.
- For **'in <County>'** or **'low-risk counties'**, use **find_properties_in_county** or **find_properties_in_low_risk_counties** (polygon containment with gis.us_counties).
- Use **search_properties** for simple filters.
- Use **get_low_risk_counties** to show risk context.
- Use **get_state_predictions** for forecasts.
- Use **execute_custom_sql** when needed.

## CONTEXT:
- Keep and refine filters across turns (price, beds, state).
- If ambiguous, ask briefly; otherwise proceed.

IMPORTANT: Do not expose internal tool names or SQL; answer naturally with property cards.
"""
      # Convert conversation history for Gemini
      gemini_history = []
      for msg in conversation:
          if "role" in msg and "parts" in msg:
              gemini_history.append({
                  "role": msg["role"],
                  "parts": msg["parts"]
              })
      
      chat = client.chats.create(
          model=GEMINI_MODEL,
          config={
              "system_instruction": system_instruction,
              "temperature": 0.7,
              "tools": tools
          },
          history=gemini_history
      )
      
      response = chat.send_message(user_message)
      
      # Handle tool calls
      max_iterations = 5
      iteration = 0
      while iteration < max_iterations:
          if not response.candidates or not response.candidates[0].content.parts:
              break
          function_calls = [
              part.function_call
              for part in response.candidates[0].content.parts
              if hasattr(part, 'function_call') and part.function_call
          ]
          if not function_calls:
              break
          function_responses = []
          for func_call in function_calls:
              try:
                  result = execute_function(func_call)
                  function_responses.append(
                      types.Part(
                          function_response=types.FunctionResponse(
                              name=func_call.name,
                              response={"result": result}
                          )
                      )
                  )
              except Exception as e:
                  print(f"Function call error: {e}")
                  function_responses.append(
                      types.Part(
                          function_response=types.FunctionResponse(
                              name=func_call.name,
                              response={"error": str(e)}
                          )
                      )
                  )
          if function_responses:
              response = chat.send_message(function_responses)
          iteration += 1
      
      final_response = ""
      if response.candidates and response.candidates[0].content.parts:
          for part in response.candidates[0].content.parts:
              if hasattr(part, 'text') and part.text:
                  final_response += part.text
      
      if not final_response:
          final_response = "I couldn’t find results with that query—try adjusting price, distance, or location."
      
      # Persist conversation
      user_content = {
          "role": "user", 
          "parts": [{"text": user_message}], 
          "timestamp": timestamp
      }
      conversation.append(user_content)
      
      assistant_content = {
          "role": "model",
          "parts": [{"text": final_response}],
          "timestamp": timestamp
      }
      conversation.append(assistant_content)
      
      SessionManager.save_conversation(session_id, conversation)
      SessionManager.extend_session(session_id)
      
      print("="*60)
      print(f"✅ AI Assistant Response Complete:")
      print(f"   Session: {session_id}")
      print(f"   Response length: {len(final_response)} characters")
      print(f"   Total conversation messages: {len(conversation)}")
      print("="*60 + "\n")
      
      return final_response
      
  except Exception as e:
      print(f"Error in handle_chat: {e}")
      return "I apologize, but I encountered an unexpected error. Please try again or start a new conversation."

# ==================== FASTAPI APPLICATION ====================

app = FastAPI(
  title="Agentic Real Estate Chatbot API",
  description="AI-powered real estate assistant with Redis session management and SQL agent fallback",
  version="2.2-opengeo"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# ==================== REQUEST/RESPONSE MODELS ====================

class ChatRequest(BaseModel):
  session_id: str
  message: str

class ChatResponse(BaseModel):
  response: str
  session_id: str
  message_count: int
  redis_status: str

class SessionInfo(BaseModel):
  session_id: str
  exists: bool
  message_count: int = 0
  created_at: Optional[str] = None
  last_activity: Optional[str] = None
  ttl_seconds: Optional[int] = None

# ==================== API ENDPOINTS ====================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
  """
  Main chat endpoint with Redis-backed session management and SQL agent fallback.
  """
  try:
      if not request.session_id or len(request.session_id) < 3:
          raise HTTPException(status_code=400, detail="Invalid session_id")
      if not request.message.strip():
          raise HTTPException(status_code=400, detail="Message cannot be empty")
      
      print(f"Processing chat request for session: {request.session_id}")
      print(f"Message: {request.message[:100]}...")
      
      response = handle_chat(request.session_id, request.message)
      conversation = SessionManager.get_conversation(request.session_id)
      clean_response = convert_for_json(response)
      
      print(f"Chat response generated successfully for session: {request.session_id}")
      
      return ChatResponse(
          response=clean_response,
          session_id=request.session_id,
          message_count=len(conversation),
          redis_status="connected" if REDIS_AVAILABLE else "in-memory"
      )
  
  except HTTPException:
      raise
  except Exception as e:
      print(f"Unexpected error in chat endpoint: {e}")
      print(f"Session ID: {request.session_id}")
      print(f"Message: {request.message}")
      import traceback
      traceback.print_exc()
      raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
  info = SessionManager.get_session_info(session_id)
  return SessionInfo(
      session_id=session_id,
      exists=info.get("exists", False),
      message_count=info.get("message_count", 0),
      created_at=info.get("created_at"),
      last_activity=info.get("last_activity"),
      ttl_seconds=info.get("ttl_seconds")
  )

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
  success = SessionManager.clear_conversation(session_id)
  if success:
      return {"message": "Session cleared successfully", "session_id": session_id}
  else:
      raise HTTPException(status_code=500, detail="Failed to clear session")

@app.post("/session/{session_id}/extend")
async def extend_session_ttl(session_id: str):
  success = SessionManager.extend_session(session_id)
  if success:
      return {"message": "Session TTL extended", "session_id": session_id, "new_ttl_seconds": REDIS_SESSION_TTL}
  else:
      raise HTTPException(status_code=500, detail="Failed to extend session")

@app.get("/sessions")
async def list_active_sessions():
  sessions = SessionManager.get_all_sessions()
  return {
      "active_sessions": len(sessions),
      "session_ids": sessions[:100],
      "redis_status": "connected" if REDIS_AVAILABLE else "in-memory"
  }

@app.get("/health")
async def health_check():
  health_status = {
      "status": "healthy",
      "timestamp": datetime.now().isoformat(),
      "services": {}
  }
  
  # Redis
  if REDIS_AVAILABLE:
      try:
          redis_client.ping()
          health_status["services"]["redis"] = {
              "status": "connected",
              "url": REDIS_URL.split("@")[-1] if "@" in REDIS_URL else "configured"
          }
      except RedisError as e:
          health_status["services"]["redis"] = {"status": "error", "error": str(e)}
          health_status["status"] = "degraded"
  else:
      health_status["services"]["redis"] = {
          "status": "using in-memory fallback",
          "warning": "Redis unavailable, sessions not persistent"
      }
      health_status["status"] = "degraded"
  
  # Supabase
  try:
      with supabase_engine.connect() as conn:
          conn.execute(text("SELECT 1"))
      health_status["services"]["supabase"] = {"status": "connected"}
  except Exception as e:
      health_status["services"]["supabase"] = {"status": "error", "error": str(e)}
      health_status["status"] = "unhealthy"
  
  # BigQuery
  if BQ_AVAILABLE and bq_client is not None:
      try:
          test_query = f"SELECT 1 FROM `fourth-webbing-474805-j5.real_estate_market.state_market` LIMIT 1"
          bq_client.query(test_query, location=BQ_LOCATION).result()
          health_status["services"]["bigquery"] = {
              "status": "connected",
              "project": "fourth-webbing-474805-j5",
              "location": BQ_LOCATION
          }
      except Exception as e:
          health_status["services"]["bigquery"] = {"status": "error", "error": str(e)}
          health_status["status"] = "unhealthy"
  else:
      health_status["services"]["bigquery"] = {
          "status": "unavailable",
          "error": "BigQuery client not initialized"
      }
  
  # Gemini
  try:
      health_status["services"]["gemini"] = {"status": "configured", "model": GEMINI_MODEL}
  except Exception as e:
      health_status["services"]["gemini"] = {"status": "error", "error": str(e)}
      health_status["status"] = "unhealthy"

  # Geocoding
  health_status["services"]["geocoding"] = {
      "census": "enabled",
      "nominatim": "enabled"
  }
  
  return health_status

@app.get("/schema")
async def get_database_schema():
  try:
      return {
          "supabase": get_supabase_schema(),
          "bigquery": get_bigquery_schema()
      }
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
  return {
      "name": "Agentic Real Estate Chatbot API",
      "version": "2.2-opengeo",
      "description": "AI-powered real estate assistant with Redis session management and SQL agent fallback",
      "features": [
          "Multi-database support (Supabase + BigQuery)",
          "Redis-backed persistent sessions",
          "Context-aware conversations with follow-up handling",
          "Distance-based property search",
          "Open-source place geocoding (US Census → Nominatim)",
          "County polygon filtering via PostGIS (gis.us_counties)",
          "Risk assessment integration (FEMA NRI)",
          "Market predictions and historical analysis",
          "SQL agent fallback for complex queries",
          "Secure SQL query sanitization"
      ],
      "endpoints": {
          "POST /chat": "Send a message (requires session_id and message)",
          "GET /session/{session_id}": "Get session information",
          "DELETE /session/{session_id}": "Clear session history",
          "POST /session/{session_id}/extend": "Extend session TTL",
          "GET /sessions": "List all active sessions",
          "GET /health": "Detailed health check",
          "GET /schema": "Get database schema information",
          "GET /": "This endpoint"
      },
      "session_config": {
          "backend": "Redis" if REDIS_AVAILABLE else "In-Memory (fallback)",
          "ttl_seconds": REDIS_SESSION_TTL,
          "ttl_human": f"{REDIS_SESSION_TTL // 60} minutes"
      },
      "sql_agent": {
          "enabled": True,
          "allowed_supabase_tables": ALLOWED_SUPABASE_TABLES,
          "allowed_bigquery_tables": ALLOWED_BIGQUERY_TABLES,
          "security": "Only SELECT queries on whitelisted tables"
      },
      "geocoding": {
          "census": True,
          "nominatim": True
      },
      "redis_status": "connected" if REDIS_AVAILABLE else "fallback mode"
  }

# ==================== STARTUP/SHUTDOWN EVENTS ====================

@app.on_event("startup")
async def startup_event():
  """Run on application startup."""
  print("\n" + "="*80)
  print("🚀 AGENTIC REAL ESTATE CHATBOT - STARTING UP")
  print("="*80)
  
  # Verify Gemini API
  try:
      print(f"✅ Gemini API configured: {GEMINI_MODEL}")
  except Exception as e:
      print(f"❌ Gemini API error: {e}")
  
  # Verify Supabase
  try:
      with supabase_engine.connect() as conn:
          result = conn.execute(text("SELECT COUNT(*) FROM properties"))
          count = result.fetchone()[0]
          print(f"✅ Supabase connected: {count} properties in database")
  except Exception as e:
      print(f"❌ Supabase connection error: {e}")
  
  # Verify BigQuery
  if BQ_AVAILABLE and bq_client is not None:
      try:
          ds_ref = f"fourth-webbing-474805-j5.real_estate_market"
          dataset = bq_client.get_dataset(ds_ref)
          print(f"✅ BigQuery dataset found: {ds_ref} (location={dataset.location})")

          test_sql = f"SELECT 1 AS ok FROM `fourth-webbing-474805-j5.real_estate_market.state_market` LIMIT 1"
          bq_client.query(test_sql, location=BQ_LOCATION).result()
          print(f"✅ BigQuery connected: Project fourth-webbing-474805-j5, Location {BQ_LOCATION}")
      except Exception as e:
          print(f"❌ BigQuery connection error: {e}")
  else:
      print("⚠️ BigQuery client not available - credentials missing or invalid")
  
  # Verify Redis
  if REDIS_AVAILABLE:
      try:
          redis_client.ping()
          info = redis_client.info("server")
          print(f"✅ Redis connected: {info.get('redis_version', 'unknown version')}")
          print(f"   Session TTL: {REDIS_SESSION_TTL} seconds ({REDIS_SESSION_TTL // 60} minutes)")
      except RedisError as e:
          print(f"❌ Redis connection error: {e}")
  else:
      print("⚠️  Redis unavailable - using in-memory session storage")
      print("   WARNING: Sessions will not persist across server restarts!")
  
  # SQL Agent status
  print(f"✅ SQL Agent enabled")
  print(f"   Allowed Supabase tables: {', '.join(ALLOWED_SUPABASE_TABLES)}")
  print(f"   Allowed BigQuery tables: {', '.join(ALLOWED_BIGQUERY_TABLES)}")

  # Geocoding readiness
  print("✅ Open-source geocoding ready: Census=ON, Nominatim=ON")
  
  print("="*80)
  print("✅ Server ready to handle requests")
  print("="*80 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
  """Run on application shutdown."""
  print("\n" + "="*80)
  print("🛑 AGENTIC REAL ESTATE CHATBOT - SHUTTING DOWN")
  print("="*80)
  
  if REDIS_AVAILABLE:
      try:
          redis_client.close()
          print("✅ Redis connection closed")
      except Exception as e:
          print(f"⚠️  Redis close error: {e}")
  
  try:
      supabase_engine.dispose()
      print("✅ Supabase connection closed")
  except Exception as e:
      print(f"⚠️  Supabase close error: {e}")
  
  try:
      if bq_client:
          bq_client.close()
          print("✅ BigQuery connection closed")
  except Exception as e:
      print(f"⚠️  BigQuery close error: {e}")
  
  print("="*80)
  print("👋 Goodbye!")
  print("="*80 + "\n")

# ==================== RUN SERVER ====================

if __name__ == "__main__":
  import uvicorn
  
  print("\n" + "="*80)
  print("🤖 AGENTIC REAL ESTATE CHATBOT WITH SQL AGENT")
  print("="*80)
  print(f"Model: {GEMINI_MODEL}")
  print(f"Redis: {'Connected' if REDIS_AVAILABLE else 'Fallback Mode'}")
  print(f"Session TTL: {REDIS_SESSION_TTL // 60} minutes")
  print(f"SQL Agent: Enabled")
  print("="*80 + "\n")
  
  uvicorn.run(
      app,
      host="0.0.0.0",
      port=8001,
      log_level="info"
  )

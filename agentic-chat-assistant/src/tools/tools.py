"""
Enhanced Tools - With Better Error Handling and Diagnostics
Includes fallback messages when BigQuery is unavailable
"""
import re
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal
import uuid

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from google.cloud import bigquery
from google.api_core import exceptions as gcp_exceptions
from geopy.distance import geodesic

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import structlog

from src.config import settings, ALLOWED_SUPABASE_TABLES, ALLOWED_BIGQUERY_TABLES, PROTECTED_TABLES
from src.agents.supabase_domain import SUPABASE_DOMAIN_KNOWLEDGE
from src.agents.bigquery_domain import BIGQUERY_DOMAIN_KNOWLEDGE

logger = structlog.get_logger()


# ==================== DATABASE CLIENTS ====================

class DatabaseClients:
    """Singleton for database connections"""
    
    _instance = None
    _supabase_engine = None
    _bigquery_client = None
    _bigquery_available = None
    _supabase_llm = None
    _bigquery_llm = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def supabase_engine(self):
        if self._supabase_engine is None:
            self._supabase_engine = create_engine(
                settings.supabase_url,
                pool_pre_ping=True,
                pool_size=settings.db_pool_size,
                max_overflow=settings.db_max_overflow,
                pool_recycle=settings.db_pool_recycle,
                connect_args={"options": "-c statement_timeout=30000"}
            )
            logger.info("supabase_engine_initialized")
        return self._supabase_engine
    
    def _get_available_tables(self) -> List[str]:
        """Get list of actually available tables"""
        try:
            inspector = inspect(self.supabase_engine)
            public_tables = inspector.get_table_names(schema='public')
            
            gis_tables = []
            try:
                schemas = inspector.get_schema_names()
                if 'gis' in schemas:
                    gis_tables = [f"gis.{t}" for t in inspector.get_table_names(schema='gis')]
            except:
                pass
            
            all_tables = public_tables + gis_tables
            
            available = []
            for allowed in ALLOWED_SUPABASE_TABLES:
                if '.' in allowed:
                    if allowed in all_tables:
                        available.append(allowed)
                else:
                    if allowed in public_tables:
                        available.append(allowed)
            
            logger.info("available_tables_detected", tables=available)
            return available or ['properties', 'nri_counties', 'uszips']
            
        except Exception as e:
            logger.error("table_detection_error", error=str(e))
            return ['properties', 'nri_counties', 'uszips']
    
    def _get_table_info(self) -> str:
        """Get table schema info for prompt"""
        try:
            inspector = inspect(self.supabase_engine)
            info = []
            
            for table in self._get_available_tables():
                if '.' not in table:
                    columns = inspector.get_columns(table)
                    col_info = ", ".join([f"{c['name']} ({c['type']})" for c in columns[:10]])
                    info.append(f"{table}: {col_info}")
            
            return "\n".join(info)
        except:
            return "properties, nri_counties, uszips tables available"
    
    @property
    def supabase_llm(self):
        """LLM for Supabase SQL generation"""
        if self._supabase_llm is None:
            self._supabase_llm = ChatGoogleGenerativeAI(
                model=settings.gemini_flash_model,
                temperature=0,
                google_api_key=settings.google_api_key
            )
        return self._supabase_llm
    
    def generate_supabase_sql(self, question: str) -> str:
        """Generate Supabase SQL with domain knowledge"""
        available = self._get_available_tables()
        table_info = self._get_table_info()
        
        system_message = f"""You are a PostgreSQL/PostGIS expert for real estate queries.

{SUPABASE_DOMAIN_KNOWLEDGE}

**Available Tables in Database:**
{', '.join(available)}

**Table Schema Info:**
{table_info}

**CRITICAL RULES:**
1. ONLY SELECT queries - never INSERT, UPDATE, DELETE, DROP
2. ONLY use available tables: {', '.join(available)}
3. NEVER query: {', '.join(PROTECTED_TABLES)}
4. Use LOWER() for case-insensitive matching
5. Always filter listing_status='active' for properties
6. Column names: bedrooms (not beds), county_geoid (not county_fips in properties)
7. Always add LIMIT (max 1000)

Generate ONLY the SQL query, no explanations."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Question: {question}\n\nSQL Query:")
        ])
        
        try:
            messages = prompt.format_messages(question=question)
            response = self.supabase_llm.invoke(messages)
            sql = response.content.strip()
            
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0].strip()
            
            if ";" in sql:
                sql = sql.split(";")[0].strip()
            
            logger.info("supabase_sql_generated", question=question[:100], sql=sql[:200])
            return sql
            
        except Exception as e:
            logger.error("supabase_sql_generation_error", error=str(e))
            raise
    
    @property
    def bigquery_llm(self):
        if self._bigquery_llm is None:
            self._bigquery_llm = ChatGoogleGenerativeAI(
                model=settings.gemini_flash_model,
                temperature=0,
                google_api_key=settings.google_api_key
            )
        return self._bigquery_llm
    
    def is_bigquery_available(self) -> bool:
        """Check if BigQuery is configured and accessible"""
        if self._bigquery_available is not None:
            return self._bigquery_available
        
        try:
            # Try to initialize client
            _ = self.bigquery_client
            
            # Try a simple query to verify access
            test_query = f"SELECT 1 FROM `{settings.bigquery_dataset_full}.state_lookup` LIMIT 1"
            job = self._bigquery_client.query(test_query)
            job.result()
            
            self._bigquery_available = True
            logger.info("bigquery_available_check", status="available")
            return True
            
        except Exception as e:
            self._bigquery_available = False
            logger.warning("bigquery_not_available", 
                         error=str(e),
                         error_type=type(e).__name__)
            return False
    
    @property
    def bigquery_client(self):
        if self._bigquery_client is None:
            try:
                self._bigquery_client = bigquery.Client(
                    project=settings.bq_project_id,
                    location=settings.bq_location
                )
                logger.info("bigquery_client_initialized", 
                           project=settings.bq_project_id,
                           dataset=settings.bigquery_dataset_full)
            except Exception as e:
                logger.error("bigquery_initialization_failed", 
                           error=str(e),
                           error_type=type(e).__name__)
                raise
        return self._bigquery_client
    
    def generate_bigquery_sql(self, question: str) -> str:
        """Generate BigQuery SQL with domain knowledge"""
        dataset = settings.bigquery_dataset_full
        tables_list = ', '.join(ALLOWED_BIGQUERY_TABLES)
        
        system_message = f"""You are a BigQuery SQL expert for real estate market data.

{BIGQUERY_DOMAIN_KNOWLEDGE}

**Dataset:** {dataset}

**CRITICAL RULES:**
1. ONLY SELECT queries
2. Use fully qualified names: `fourth-webbing-474805-j5.{dataset}.<table_name>` (replace <table_name> with actual table)
3. Available tables: {tables_list}
4. Standard SQL syntax
5. Always add LIMIT clause
6. JOIN with lookup tables for readable names

Example:
SELECT sm.*, sl.state 
FROM `{dataset}.state_market` sm
JOIN `{dataset}.state_lookup` sl ON sm.state_num = sl.state_num
WHERE sl.state_id = 'TX'
LIMIT 50

Generate ONLY the SQL query, no explanations."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Question: {question}\n\nSQL Query:")
        ])
        
        try:
            messages = prompt.format_messages(question=question)
            response = self.bigquery_llm.invoke(messages)
            sql = response.content.strip()
            
            if "```" in sql:
                sql = sql.split("```")[1].split("```")[0].replace("sql", "").strip()
            if ";" in sql:
                sql = sql.split(";")[0].strip()
            
            logger.info("bigquery_sql_generated", 
                       question=question[:100],
                       sql=sql[:200])
            return sql
            
        except Exception as e:
            logger.error("bigquery_sql_generation_error", 
                        error=str(e),
                        error_type=type(e).__name__)
            raise


db_clients = DatabaseClients()


# ==================== UTILITIES ====================

def convert_for_json(obj: Any) -> Any:
    """Convert to JSON-serializable"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {str(k): convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(i) for i in obj]
    return str(obj)


def validate_select_only(query: str) -> bool:
    """Strict SELECT-only validation"""
    q = query.upper().strip()
    if not q.startswith('SELECT'):
        return False
    dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
                 'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE', 'EXEC']
    for kw in dangerous:
        if re.search(r'\b' + kw + r'\b', q):
            return False
    if re.search(r'--|/\*|\*/', query):
        return False
    if re.search(r';.*(SELECT|INSERT|UPDATE|DELETE)', q):
        return False
    return True


def sanitize_sql_query(query: str, database: str = "supabase") -> Optional[str]:
    """Sanitize SQL query"""
    try:
        query = query.strip()
        if not validate_select_only(query):
            logger.warning("query_validation_failed")
            return None
        
        for protected in PROTECTED_TABLES:
            if re.search(r'\b' + protected.upper() + r'\b', query.upper()):
                logger.warning("protected_table_attempt", table=protected)
                return None
        
        if "LIMIT" not in query.upper():
            query += f" LIMIT {settings.max_query_results}"
        
        logger.info("query_sanitization_success")
        return query
    except Exception as e:
        logger.error("sanitization_error", error=str(e))
        return None


# ==================== GEOCODING ====================

def _geocode_census(place: str) -> Optional[Dict]:
    try:
        url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
        r = requests.get(url, params={"address": place, "benchmark": "Public_AR_Current", "format": "json"}, timeout=8)
        data = r.json()
        matches = data.get("result", {}).get("addressMatches", [])
        if matches:
            coords = matches[0].get("coordinates", {})
            if coords.get("y") and coords.get("x"):
                return {"lat": float(coords["y"]), "lon": float(coords["x"]), 
                       "formatted_address": matches[0].get("matchedAddress", place), "source": "census"}
    except:
        pass
    return None


def _geocode_nominatim(place: str) -> Optional[Dict]:
    try:
        url = "https://nominatim.openstreetmap.org/search"
        r = requests.get(url, params={"q": place, "format": "json", "limit": 1}, 
                        headers={"User-Agent": settings.geocoding_user_agent}, timeout=8)
        results = r.json()
        if results:
            return {"lat": float(results[0]["lat"]), "lon": float(results[0]["lon"]),
                   "formatted_address": results[0].get("display_name", place), "source": "nominatim"}
    except:
        pass
    return None


@tool
def geocode_location(place: str) -> Dict[str, Any]:
    """Geocode location to coordinates"""
    if not place:
        return {"error": "Place required", "success": False}
    result = _geocode_census(place) or _geocode_nominatim(place)
    if result:
        return {"success": True, **result}
    return {"error": f"Could not geocode: {place}", "success": False}


@tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Dict:
    """Calculate distance between coordinates"""
    try:
        miles = geodesic((lat1, lon1), (lat2, lon2)).miles
        return {"success": True, "distance_miles": round(miles, 2), 
               "distance_km": round(miles * 1.60934, 2)}
    except Exception as e:
        return {"error": str(e), "success": False}


# ==================== SQL TOOLS ====================

@tool
def query_supabase_natural_language(question: str) -> Dict[str, Any]:
    """
    Query Supabase using natural language with domain knowledge.
    Use for: property listings, risk data, location data.
    """
    logger.info("supabase_natural_language_query", question=question[:200])
    
    try:
        generated_sql = db_clients.generate_supabase_sql(question)
        safe_query = sanitize_sql_query(generated_sql)
        
        if not safe_query:
            return {"error": "Query failed validation", "success": False}
        
        with db_clients.supabase_engine.connect() as conn:
            result = conn.execute(text(safe_query))
            rows = [convert_for_json(dict(r._mapping)) for r in result.fetchall()]
            
            logger.info("query_success", count=len(rows))
            return {
                "success": True, 
                "rows": rows, 
                "count": len(rows), 
                "query": safe_query,
                "question": question
            }
            
    except Exception as e:
        logger.error("query_error", error=str(e), error_type=type(e).__name__)
        return {"error": str(e), "success": False, "question": question}


@tool
def query_bigquery_natural_language(question: str) -> Dict[str, Any]:
    """
    Query BigQuery using natural language with domain knowledge.
    Use for: market trends, statistics, predictions.
    """
    logger.info("bigquery_natural_language_query", question=question[:200])
    
    # Check if BigQuery is available
    if not db_clients.is_bigquery_available():
        logger.warning("bigquery_unavailable_fallback")
        return {
            "success": False,
            "error": "BigQuery market data is currently unavailable. Please check configuration.",
            "fallback_message": "Market data requires BigQuery access. Please verify: 1) Google Cloud credentials are configured, 2) Project ID and dataset are correct, 3) Service account has BigQuery permissions.",
            "question": question
        }
    
    try:
        generated_sql = db_clients.generate_bigquery_sql(question)
        logger.info("bigquery_sql_generated", sql=generated_sql[:300])
        
        safe_query = sanitize_sql_query(generated_sql, "bigquery")
        if not safe_query:
            return {"error": "Query failed validation", "success": False}
        
        # Execute query
        query_job = db_clients.bigquery_client.query(safe_query)
        rows = [convert_for_json(dict(r)) for r in query_job.result()]
        
        logger.info("bigquery_success", count=len(rows))
        return {
            "success": True, 
            "rows": rows, 
            "count": len(rows), 
            "query": safe_query,
            "question": question
        }
        
    except gcp_exceptions.GoogleAPIError as e:
        logger.error("bigquery_api_error", 
                    error=str(e),
                    error_code=e.code if hasattr(e, 'code') else None)
        return {
            "error": f"BigQuery API error: {str(e)}",
            "success": False,
            "fallback_message": "There was an issue accessing BigQuery. Please check your Google Cloud configuration and permissions."
        }
    except Exception as e:
        logger.error("bigquery_error", error=str(e), error_type=type(e).__name__)
        return {
            "error": str(e),
            "success": False,
            "fallback_message": "Unable to retrieve market data at this time."
        }


@tool
def query_supabase(sql_query: str) -> Dict[str, Any]:
    """Execute direct SQL on Supabase"""
    safe_query = sanitize_sql_query(sql_query)
    if not safe_query:
        return {"error": "Query failed validation", "success": False}
    try:
        with db_clients.supabase_engine.connect() as conn:
            result = conn.execute(text(safe_query))
            rows = [convert_for_json(dict(r._mapping)) for r in result.fetchall()]
            return {"success": True, "rows": rows, "count": len(rows), "query": safe_query}
    except Exception as e:
        return {"error": str(e), "success": False}


@tool
def query_bigquery(sql_query: str) -> Dict[str, Any]:
    """Execute direct SQL on BigQuery"""
    if not db_clients.is_bigquery_available():
        return {
            "success": False,
            "error": "BigQuery is not available",
            "fallback_message": "Market data queries require BigQuery configuration."
        }
    
    safe_query = sanitize_sql_query(sql_query, "bigquery")
    if not safe_query:
        return {"error": "Query failed validation", "success": False}
    try:
        job = db_clients.bigquery_client.query(safe_query)
        rows = [convert_for_json(dict(r)) for r in job.result()]
        return {"success": True, "rows": rows, "count": len(rows), "query": safe_query}
    except Exception as e:
        return {"error": str(e), "success": False}


# ==================== OTHER TOOLS ====================

@tool
def search_web(query: str, num_results: int = 5) -> Dict:
    """Search web"""
    return {"success": True, "query": query, "results": [], 
           "message": "Web search needs API integration"}


@tool
def analyze_investment_potential(location: Optional[str] = None, 
                                max_price: Optional[float] = None,
                                min_beds: Optional[int] = None) -> Dict:
    """Analyze investment potential"""
    try:
        analysis = {"success": True, "timestamp": datetime.now().isoformat(),
                   "criteria": {"location": location, "max_price": max_price, "min_beds": min_beds}}
        risk_result = query_supabase_natural_language.invoke({
            "question": "Show me counties with Very Low or Low risk_index_rating"
        })
        if risk_result.get("success"):
            analysis["low_risk_counties"] = risk_result.get("rows", [])[:10]
        return analysis
    except Exception as e:
        return {"error": str(e), "success": False}


@tool
def find_real_estate_agents(location: str) -> Dict:
    """Direct to agent finder"""
    return {"success": True, "message": "Visit agent finder portal",
           "agent_finder_url": settings.agent_finder_url, "location": location}


# Export tools
TOOLS = [
    geocode_location,
    calculate_distance,
    query_supabase_natural_language,
    query_bigquery_natural_language,
    query_supabase,
    query_bigquery,
    search_web,
    analyze_investment_potential,
    find_real_estate_agents,
]
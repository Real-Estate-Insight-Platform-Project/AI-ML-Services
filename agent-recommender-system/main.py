# main.py — Supabase REST client version with enhanced geographical support
import os, math
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, PositiveInt
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ✅ Supabase Python client imports
from supabase import create_client, Client

from scorer import AgentScorer  

# ── ENV ────────────────────────────────────────────────────────────────────────
load_dotenv()

SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in env")

# ✅ Initialize Supabase REST client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# ── APP ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Agent Recommender API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scorer = AgentScorer()

# ── STANDARDIZED LOCATIONS (City, State format) ──────────────────────────────────────────────────────
STANDARDIZED_LOCATIONS = {
    "major_markets": [
        "New York City, New York", "Los Angeles, California", "Chicago, Illinois", 
        "Dallas, Texas", "Houston, Texas", "Washington, D.C.", "Miami, Florida",
        "Philadelphia, Pennsylvania", "Atlanta, Georgia", "Phoenix, Arizona", 
        "Boston, Massachusetts", "San Francisco, California", "Detroit, Michigan", 
        "Seattle, Washington", "Minneapolis, Minnesota", "San Diego, California", 
        "Tampa, Florida", "Denver, Colorado", "St. Louis, Missouri", "Baltimore, Maryland", 
        "Charlotte, North Carolina", "Portland, Oregon", "San Antonio, Texas", 
        "Orlando, Florida", "Cincinnati, Ohio", "Cleveland, Ohio", "Kansas City, Missouri", 
        "Las Vegas, Nevada", "Columbus, Ohio", "Indianapolis, Indiana", 
        "Nashville, Tennessee", "Virginia Beach, Virginia", "Providence, Rhode Island", 
        "Milwaukee, Wisconsin", "Jacksonville, Florida", "Memphis, Tennessee", 
        "Oklahoma City, Oklahoma", "Louisville, Kentucky", "Hartford, Connecticut", 
        "Richmond, Virginia", "New Orleans, Louisiana", "Buffalo, New York", 
        "Raleigh, North Carolina", "Birmingham, Alabama", "Salt Lake City, Utah", 
        "Rochester, New York", "Grand Rapids, Michigan", "Tucson, Arizona", 
        "Tulsa, Oklahoma", "Honolulu, Hawaii", "Albany, New York", "Fresno, California", 
        "Dayton, Ohio", "Syracuse, New York", "Austin, Texas", "Fort Worth, Texas", 
        "Sacramento, California", "Long Beach, California", "Mesa, Arizona", 
        "Colorado Springs, Colorado", "Omaha, Nebraska", "Albuquerque, New Mexico", 
        "Bakersfield, California", "Stockton, California"
    ],
    "florida_markets": [
        "Miami, Florida", "Fort Lauderdale, Florida", "Tampa, Florida", "Orlando, Florida", 
        "Jacksonville, Florida", "Naples, Florida", "Fort Myers, Florida", "Cape Coral, Florida", 
        "West Palm Beach, Florida", "Gainesville, Florida", "Tallahassee, Florida", 
        "Pensacola, Florida", "Sarasota, Florida", "St Petersburg, Florida", "Clearwater, Florida", 
        "Hollywood, Florida", "Coral Springs, Florida", "Pembroke Pines, Florida", 
        "Miramar, Florida", "Davie, Florida", "Plantation, Florida", "Sunrise, Florida", 
        "Pompano Beach, Florida", "Deerfield Beach, Florida", "Boca Raton, Florida", 
        "Delray Beach, Florida", "Boynton Beach, Florida", "Lake Worth, Florida", 
        "Bonita Springs, Florida", "Marco Island, Florida", "Estero, Florida", 
        "Lehigh Acres, Florida", "North Fort Myers, Florida", "East Fort Myers, Florida", 
        "Fort Myers Beach, Florida", "Punta Gorda, Florida", "Port Charlotte, Florida", 
        "Venice, Florida", "Bradenton, Florida", "Lakeland, Florida", "Winter Haven, Florida", 
        "Kissimmee, Florida", "Sanford, Florida", "Deltona, Florida", "Palm Bay, Florida", 
        "Melbourne, Florida", "Cocoa, Florida", "Titusville, Florida", "Daytona Beach, Florida", 
        "Ocala, Florida", "The Villages, Florida"
    ],
    "states": [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
        "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", 
        "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
        "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", 
        "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
        "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
        "Wisconsin", "Wyoming", "District of Columbia"
    ]
}

# ── MODELS ────────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    locations: Optional[List[str]] = None  # Changed from 'regions' to 'locations' for clarity
    property_types: Optional[List[str]] = None
    price_min: Optional[float] = Field(default=None, ge=0)
    price_max: Optional[float] = Field(default=None, ge=0)
    top_k: PositiveInt = 10
    min_rating: float = 0.0
    min_reviews: int = 0
    require_phone: bool = False

class AgentOut(BaseModel):
    rank: int
    agent_id: int
    name: str
    brokerage: Optional[str]
    star_rating: float
    num_reviews: int
    past_year_deals: int
    avg_transaction_value: float
    service_regions: List[str]
    comprehensive_areas: List[str]  # New field for enhanced geographical data
    business_market: Optional[str]  # New field
    office_state: Optional[str]     # New field
    property_types: List[str]
    phone: Optional[str]
    email: Optional[str]
    profile_url: Optional[str]
    is_premier: bool
    is_active: bool
    total_score: float
    breakdown: Dict[str, Any]

class RecommendResponse(BaseModel):
    total_matches: int
    returned: int
    results: List[AgentOut]

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def get_available_locations():
    """Get all standardized locations that users can search for."""
    return STANDARDIZED_LOCATIONS

def _normalize_user_locations(locations: List[str]) -> List[str]:
    """Normalize and validate user-provided locations to "City, State" format."""
    if not locations:
        return []
    
    normalized = []
    all_locations = (
        STANDARDIZED_LOCATIONS["major_markets"] + 
        STANDARDIZED_LOCATIONS["florida_markets"] + 
        STANDARDIZED_LOCATIONS["states"]
    )
    all_locations_lower = [loc.lower() for loc in all_locations]
    
    for loc in locations:
        loc_clean = str(loc).strip()
        if not loc_clean:
            continue
            
        # Direct match (case insensitive)
        loc_lower = loc_clean.lower()
        if loc_lower in all_locations_lower:
            idx = all_locations_lower.index(loc_lower)
            normalized.append(all_locations[idx])
            continue
        
        # Handle old format inputs and partial matches
        best_match = None
        best_score = 0
        
        for standard_loc in all_locations:
            standard_lower = standard_loc.lower()
            
            # Extract city and state from "City, State" format
            if ", " in standard_loc:
                city, state = [part.strip() for part in standard_loc.split(", ", 1)]
            else:
                city, state = standard_loc, ""
            
            # Check various matching patterns
            if loc_lower == standard_lower:
                # Exact match
                best_match = standard_loc
                best_score = 100
                break
            elif loc_lower == city.lower():
                # City name match
                if best_score < 90:
                    best_match = standard_loc
                    best_score = 90
            elif loc_lower == state.lower():
                # State name match
                if best_score < 80:
                    best_match = standard_loc
                    best_score = 80
            elif loc_lower in standard_lower or standard_lower in loc_lower:
                # Partial match
                if best_score < 70:
                    best_match = standard_loc
                    best_score = 70
            
            # Handle abbreviation to full state conversion
            state_abbrevs = {
                "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
                "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
                "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
                "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
                "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
                "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
                "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
                "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
                "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
                "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
                "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
                "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
                "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia"
            }
            
            if loc_lower in state_abbrevs and state_abbrevs[loc_lower] == state.lower():
                if best_score < 85:
                    best_match = standard_loc
                    best_score = 85
        
        if best_match and best_score >= 70:
            normalized.append(best_match)
        else:
            # If no good match found, add as-is for flexibility
            normalized.append(loc_clean)
    
    return list(set(normalized))  # Remove duplicates

def _extract_location_components(location: str) -> tuple[str, str]:
    """Extract city and state from 'City, State' format."""
    if ", " in location:
        parts = location.split(", ", 1)
        return parts[0].strip(), parts[1].strip()
    else:
        return location.strip(), ""

def _extract_location_components(location: str) -> tuple[str, str]:
    """Extract city and state from 'City, State' format."""
    if ", " in location:
        parts = location.split(", ", 1)
        return parts[0].strip(), parts[1].strip()
    else:
        return location.strip(), ""

# ── DB HELPERS (Supabase REST) ────────────────────────────────────────────────
def _fetch_candidates_from_db(req: RecommendRequest, page_size: int = 1000) -> pd.DataFrame:
    """
    Fetch candidates with Supabase client.
    Enhanced to use comprehensive geographical filtering.
    """
    q = supabase.table("agents").select("*", count="exact").eq("is_active", True)

    if req.min_rating and req.min_rating > 0:
        q = q.gte("star_rating", req.min_rating)

    if req.min_reviews and req.min_reviews > 0:
        q = q.gte("num_reviews", req.min_reviews)

    if req.require_phone:
        q = q.neq("phone_number", "Not Available")

    # Enhanced geographical filtering using multiple fields
    if req.locations:
        normalized_locations = _normalize_user_locations(req.locations)
        if normalized_locations:
            # Create OR conditions for different geographical fields
            # Note: Supabase Python client might need multiple queries for complex OR conditions
            # For now, we'll use comprehensive_service_areas as primary filter
            q = q.overlaps("comprehensive_service_areas", normalized_locations)

    # Property types filtering
    if req.property_types:
        q = q.overlaps("property_types", req.property_types)

    # Soft price prefilter (let scoring handle exactness)
    if (req.price_min is not None) and (req.price_max is not None):
        soft_min = req.price_min * 0.5
        soft_max = req.price_max * 2.0
        q = q.gte("deal_prices_median", soft_min).lte("deal_prices_median", soft_max)

    # First page (+count)
    resp = q.order("past_year_deals", desc=True).order("star_rating", desc=True).range(0, page_size - 1).execute()
    if getattr(resp, "error", None):
        raise HTTPException(status_code=500, detail=f"Supabase error: {resp.error}")

    total = resp.count or 0
    rows = resp.data or []

    # Paginate if needed
    pages = math.ceil(max(0, total - len(rows)) / page_size)
    for i in range(pages):
        start = (i + 1) * page_size
        end = start + page_size - 1
        nxt = q.range(start, end).execute()
        if getattr(nxt, "error", None):
            raise HTTPException(status_code=500, detail=f"Supabase error: {nxt.error}")
        rows.extend(nxt.data or [])

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ensure numeric types
    numeric_cols = [
        "star_rating","num_reviews","past_year_deals","business_market_id",
        "home_transactions_lifetime","transaction_volume_lifetime",
        "deal_prices_median","deal_prices_min","deal_prices_max",
        "deal_prices_std","avg_transaction_value","num_comprehensive_areas"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Handle array fields - convert from Postgres text[] format back to Python lists
    array_cols = ["primary_service_regions", "property_types", "comprehensive_service_areas"]
    for c in array_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: _parse_pg_text_array(v) if isinstance(v, str) else (v if isinstance(v, list) else []))

    # Booleans
    bool_cols = ["partner","is_premier","serves_offers","serves_listings","is_active","profile_contact_enabled"]
    for b in bool_cols:
        if b in df.columns:
            df[b] = df[b].astype(bool)

    return df

def _parse_pg_text_array(text_array: str) -> List[str]:
    """Parse Postgres text[] format back to Python list."""
    if not text_array or text_array == "{}":
        return []
    
    # Remove outer braces and split by comma
    inner = text_array.strip("{}")
    if not inner:
        return []
    
    # Simple parsing - assumes no commas within quoted strings for now
    items = []
    for item in inner.split(","):
        item = item.strip().strip('"')
        if item:
            items.append(item)
    
    return items

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/locations")
def get_locations():
    """Get all available standardized locations for frontend dropdown/autocomplete."""
    return get_available_locations()

@app.get("/debug/count")
def debug_count():
    r = supabase.table("agents").select("agent_id", count="exact").execute()
    return {"count": r.count, "error": getattr(r, "error", None)}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if (req.price_min is not None) ^ (req.price_max is not None):
        raise HTTPException(400, "Provide both price_min and price_max or neither.")

    df = _fetch_candidates_from_db(req)
    if df.empty:
        return RecommendResponse(total_matches=0, returned=0, results=[])

    # Normalize user locations for scoring
    normalized_locations = _normalize_user_locations(req.locations) if req.locations else None
    
    price_range: Optional[Tuple[float, float]] = (
        (req.price_min, req.price_max) if req.price_min is not None else None
    )
    
    # Use normalized locations for scoring
    total, breakdown = scorer.total(df, normalized_locations, req.property_types, price_range)

    out = df.copy().reset_index(drop=True)
    out[breakdown.columns] = breakdown
    out = out.sort_values("total_score", ascending=False).head(req.top_k)

    results: List[AgentOut] = []
    for i, r in enumerate(out.itertuples(index=False), 1):
        results.append(AgentOut(
            rank=i,
            agent_id=int(getattr(r, "agent_id")),
            name=getattr(r, "name"),
            brokerage=getattr(r, "brokerage_name"),
            star_rating=float(getattr(r, "star_rating")),
            num_reviews=int(getattr(r, "num_reviews")),
            past_year_deals=int(getattr(r, "past_year_deals")),
            avg_transaction_value=float(getattr(r, "avg_transaction_value")),
            service_regions=list(getattr(r, "primary_service_regions") or []),
            comprehensive_areas=list(getattr(r, "comprehensive_service_areas") or []),
            business_market=getattr(r, "business_market_normalized"),
            office_state=getattr(r, "office_state"),
            property_types=list(getattr(r, "property_types") or []),
            phone=getattr(r, "phone_number"),
            email=getattr(r, "email"),
            profile_url=getattr(r, "profile_url"),
            is_premier=bool(getattr(r, "is_premier")),
            is_active=bool(getattr(r, "is_active")),
            total_score=float(getattr(r, "total_score")),
            breakdown={
                "performance": float(getattr(r, "performance_score")),
                "market_expertise": float(getattr(r, "expertise_score")),
                "client_satisfaction": float(getattr(r, "satisfaction_score")),
                "professional_standing": float(getattr(r, "professional_score")),
                "availability": float(getattr(r, "availability_score")),
            }
        ))

    return RecommendResponse(
        total_matches=len(df), 
        returned=len(results), 
        results=results
    )

# Run locally:
#   uvicorn main:app --reload --port 8002
# main.py  — Supabase REST client version (no psycopg)
import os, math
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, PositiveInt
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ✅ Supabase Python client imports (missing before)
from supabase import create_client, Client  # docs show create_client() init

from scorer import AgentScorer  

# ── ENV ────────────────────────────────────────────────────────────────────────
load_dotenv()

SUPABASE_URL  = os.getenv("SUPABASE_URL")           # e.g. https://<project-ref>.supabase.co
SUPABASE_KEY  = os.getenv("SUPABASE_SERVICE_KEY")   # service role key (server-side only)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in env")

# ✅ Initialize Supabase REST client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# ── APP ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Agent Recommender API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],  # see FastAPI CORS docs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scorer = AgentScorer()

# ── MODELS ────────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    regions: Optional[List[str]] = None
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

# ── DB HELPERS (Supabase REST) ────────────────────────────────────────────────
def _fetch_candidates_from_db(req: RecommendRequest, page_size: int = 1000) -> pd.DataFrame:
    """
    Fetch candidates with Supabase client.
    Uses .overlaps() for text[] array filters and paginates until all rows fetched.
    """
    q = supabase.table("agents").select("*", count="exact").eq("is_active", True)

    if req.min_rating and req.min_rating > 0:
        q = q.gte("star_rating", req.min_rating)

    if req.min_reviews and req.min_reviews > 0:
        q = q.gte("num_reviews", req.min_reviews)

    if req.require_phone:
        q = q.neq("phone_number", "Not Available")

    # Array overlaps: match if there is any common element (&&)
    if req.regions:
        q = q.overlaps("primary_service_regions", req.regions)  # official .overlaps()
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
    for c in [
        "star_rating","num_reviews","past_year_deals",
        "home_transactions_lifetime","transaction_volume_lifetime",
        "deal_prices_median","deal_prices_min","deal_prices_max",
        "deal_prices_std","avg_transaction_value"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Arrays arrive as Python lists; normalize just in case
    for c in ["primary_service_regions","property_types"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [str(v)]))

    # Booleans
    for b in ["partner","is_premier","serves_offers","serves_listings","is_active","profile_contact_enabled"]:
        if b in df.columns:
            df[b] = df[b].astype(bool)

    return df

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

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

    price_range: Optional[Tuple[float, float]] = (
        (req.price_min, req.price_max) if req.price_min is not None else None
    )
    total, breakdown = scorer.total(df, req.regions, req.property_types, price_range)

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

    return RecommendResponse(total_matches=len(df), returned=len(results), results=results)

# Run locally:
#   uvicorn main:app --reload --port 8002

# risk-map/app/main.py

import os
import time
from typing import Optional, Tuple, Any, Dict, List

import orjson
import asyncpg
from fastapi import FastAPI, Query, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response, JSONResponse
from dotenv import load_dotenv

load_dotenv()

DB_URL    = os.environ["MAP_DATABASE_URL"]
# GEO_PATH = os.environ.get("COUNTIES_GEOJSON", "data/counties.min.geojson")  
# CACHE_TTL = int(os.getenv("NRI_GEOJSON_TTL", "86400"))  # seconds

# ---------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------

app = FastAPI(title="Risk Map API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress large responses (useful for tiles or others)
app.add_middleware(GZipMiddleware, minimum_size=1500)

# ---------------------------------------------------------------------
# Startup / shutdown: db pool
# ---------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    app.state.pool = await asyncpg.create_pool(
        DB_URL,
        min_size=1,
        max_size=6,
        statement_cache_size=0,  # PgBouncer safe
        max_inactive_connection_lifetime=60.0,
    )

@app.on_event("shutdown")
async def _shutdown():
    pool = getattr(app.state, "pool", None)
    if pool:
        await pool.close()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    # fallback, e.g. Decimal
    try:
        # for Decimal or others
        return float(obj)
    except Exception:
        return str(obj)

# ---------------------------------------------------------------------
# Data accessors
# ---------------------------------------------------------------------

async def fetch_all_attrs() -> Dict[str, Dict[str, Any]]:
    sql = """
      SELECT
        county_fips,
        county_name,
        state_name,
        state_fips,
        risk_index_score,
        risk_index_rating,
        predominant_hazard
      FROM public.nri_counties
    """
    pool: asyncpg.Pool = app.state.pool
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql)

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        fips = str(r["county_fips"]).zfill(5)
        out[fips] = {
            "county_fips": fips,
            "county_name": r["county_name"],
            "state_name": r["state_name"],
            "state_fips": str(r["state_fips"]).zfill(2) if r["state_fips"] is not None else None,
            "risk_index_score": float(r["risk_index_score"]) if r["risk_index_score"] is not None else None,
            "risk_index_rating": r["risk_index_rating"],
            "predominant_hazard": r["predominant_hazard"],
        }
    return out

async def fetch_one_county(fips: str) -> Optional[Dict[str, Any]]:
    sql = """
      SELECT
        county_fips,
        county_name,
        state_name,
        state_fips,
        risk_index_score,
        risk_index_rating,
        predominant_hazard
      FROM public.nri_counties
      WHERE county_fips = $1
    """
    pool: asyncpg.Pool = app.state.pool
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, fips)
    if not row:
        return None
    return {
        "fips": str(row["county_fips"]).zfill(5),
        "county_name": row["county_name"],
        "state_name": row["state_name"],
        "state_fips": str(row["state_fips"]).zfill(2) if row["state_fips"] is not None else None,
        "nri_composite_score": float(row["risk_index_score"]) if row["risk_index_score"] is not None else None,
        "nri_composite_rating": row["risk_index_rating"],
        "predominant_hazard": row["predominant_hazard"],
    }

# ---------------------------------------------------------------------
# Tile endpoint for counties - vector tile
# ---------------------------------------------------------------------

@app.get("/tiles/counties/{z}/{x}/{y}")
async def tiles_counties(z: int, x: int, y: int):
    sql = "SELECT gis.us_counties_mvt($1,$2,$3) AS mvt"
    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(sql, z, x, y)
    data = row["mvt"] if row and row["mvt"] else None
    return Response(
      content=bytes(data) if data else b"",
      media_type="application/vnd.mapbox-vector-tile",
      headers={"Cache-Control": "public, max-age=3600, s-maxage=3600"},
      status_code=200 if data else 204
    )



# ---------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------

@app.get("/", tags=["health"])
async def root():
    return {"ok": True, "service": "risk-map", "message": "Vector tile version active"}


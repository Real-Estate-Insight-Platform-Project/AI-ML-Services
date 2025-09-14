# risk-map/app/main.py
import os, json, time, asyncio
from decimal import Decimal
from typing import Any, Dict, Optional, List

import asyncpg
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

DB_URL    = os.environ["DATABASE_URL"]
GEO_PATH  = os.environ.get("COUNTIES_GEOJSON", "data/counties.min.geojson")
CACHE_TTL = int(os.getenv("NRI_GEOJSON_TTL", "86400"))  # seconds

app = FastAPI(title="Risk Map API")

# CORS: tighten allow_origins to your deployed Next.js domains later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- helpers ----------------
def _to_jsonable(obj: Any) -> Any:
    """Recursively convert Decimals etc. to JSON-safe values."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj

def _parse_hazards(h: Any) -> Dict[str, Any]:
    """
    Ensure hazards is a dict. Supabase CSV imports sometimes store JSON as text.
    This accepts dict/list/str and returns a dict safely.
    """
    if h is None:
        return {}
    if isinstance(h, dict):
        return h
    if isinstance(h, list):
        # uncommon, but make it a dict with indices
        return {str(i): v for i, v in enumerate(h)}
    if isinstance(h, str):
        try:
            j = json.loads(h)
            if isinstance(j, dict):
                return j
            if isinstance(j, list):
                return {str(i): v for i, v in enumerate(j)}
            return {}
        except Exception:
            return {}
    return {}

def _feature_in_bbox(geom: Dict[str, Any], bbox: Optional[List[float]]) -> bool:
    if not bbox:
        return True
    minx, miny, maxx, maxy = bbox

    def iter_coords(g):
        t = g["type"]
        if t == "Polygon":
            for ring in g["coordinates"]:
                for x, y in ring:
                    yield x, y
        elif t == "MultiPolygon":
            for poly in g["coordinates"]:
                for ring in poly:
                    for x, y in ring:
                        yield x, y

    xs, ys = zip(*iter_coords(geom))
    gxmin, gxmax = min(xs), max(xs)
    gymin, gymax = min(ys), max(ys)
    return not (gxmax < minx or gxmin > maxx or gymax < miny or gymax > maxy)

HAZARD_ORDER = [
    "Avalanche","Coastal Flooding","Cold Wave","Drought","Earthquake","Hail",
    "Heat Wave","Hurricane","Ice Storm","Landslide","Lightning","Riverine Flooding",
    "Strong Wind","Tornado","Tsunami","Volcanic Activity","Wildfire","Winter Weather"
]

# ---------------- load polygons once ----------------
with open(GEO_PATH, "r", encoding="utf-8") as f:
    POLY = json.load(f)  # FeatureCollection with properties.GEOID/STATEFP/NAME

# ---------------- connection pool ----------------
@app.on_event("startup")
async def _startup():
    # Disable prepared statement cache for PgBouncer
    app.state.pool = await asyncpg.create_pool(
        DB_URL,
        min_size=1,
        max_size=6,
        statement_cache_size=0,                 # <-- IMPORTANT
        max_inactive_connection_lifetime=60.0,  # optional hygiene
    )

@app.on_event("shutdown")
async def _shutdown():
    pool = getattr(app.state, "pool", None)
    if pool:
        await pool.close()

# ---------------- DB accessors ----------------
async def fetch_all_attrs() -> Dict[str, Dict[str, Any]]:
    sql = """
      select county_fips, county_name, state_name, state_fips,
             risk_index_score, risk_index_rating, hazards
      from public.nri_counties
    """
    pool: asyncpg.pool.Pool = app.state.pool
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql)

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        fips = str(r["county_fips"]).zfill(5)
        out[fips] = {
            "county_fips": fips,
            "county_name": r["county_name"],
            "state_name":  r["state_name"],
            "state_fips":  str(r["state_fips"]).zfill(2) if r["state_fips"] else None,
            "risk_index_score": r["risk_index_score"],
            "risk_index_rating": r["risk_index_rating"],
            "hazards": _parse_hazards(r["hazards"]),
        }
    return out

async def fetch_one_county(fips: str) -> Optional[Dict[str, Any]]:
    sql = """
      select county_fips, county_name, state_name, state_fips,
             risk_index_score, risk_index_rating, hazards
      from public.nri_counties
      where county_fips = $1
    """
    pool: asyncpg.pool.Pool = app.state.pool
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, fips)
    if not row:
        return None
    return {
        "fips": str(row["county_fips"]).zfill(5),
        "county_name": row["county_name"],
        "state_name": row["state_name"],
        "state_fips": str(row["state_fips"]).zfill(2) if row["state_fips"] else None,
        "nri_composite_score": row["risk_index_score"],
        "nri_composite_rating": row["risk_index_rating"],
        "hazards": _parse_hazards(row["hazards"]),
    }

# ---------------- join polygons + attributes ----------------
def join_geo(attrs: Dict[str, Dict[str, Any]],
             state_fips: Optional[str] = None,
             bbox: Optional[List[float]] = None) -> Dict[str, Any]:
    fc = {"type": "FeatureCollection", "features": []}
    for feat in POLY.get("features", []):
        props = feat.get("properties") or {}
        geoid = str(props.get("GEOID") or props.get("geoid") or "").zfill(5)
        if not geoid:
            continue

        stfp = str(props.get("STATEFP") or "").zfill(2)
        if state_fips and stfp != state_fips.zfill(2):
            continue
        if not _feature_in_bbox(feat["geometry"], bbox):
            continue

        a = attrs.get(geoid)
        merged = {
            "county_fips": geoid,
            "NAME": props.get("NAME"),
            "STATEFP": stfp,
        }
        if a:
            merged.update({
                "county_name": a.get("county_name") or props.get("NAME"),
                "state_name":  a.get("state_name"),
                "state_fips":  a.get("state_fips"),
                "risk_rating_composite": a.get("risk_index_rating"),
                "risk_score_composite":  a.get("risk_index_score"),
                "hazards": a.get("hazards"),
            })

        fc["features"].append({
            "type": "Feature",
            "geometry": feat["geometry"],
            "properties": merged
        })
    return fc

# simple in-memory cache for full-US response
_full_cache: Dict[str, Any] = {"ts": 0, "fc": None}

# ---------------- routes ----------------
@app.get("/", tags=["health"])
async def root():
    return {"ok": True, "service": "risk-map", "polygons": len(POLY.get("features", []))}

@app.get("/risk/counties.geojson")
async def counties_geojson(
    state_fips: Optional[str] = Query(default=None, description="2-digit state FIPS"),
    bbox: Optional[str]     = Query(default=None, description="minx,miny,maxx,maxy (lon/lat)")
):
    # cache only the full-US, no-filter variant
    now = time.time()
    if not state_fips and not bbox and _full_cache["fc"] and now - _full_cache["ts"] < CACHE_TTL:
        return Response(
            content=json.dumps(_full_cache["fc"]),
            media_type="application/geo+json"
        )

    attrs = await fetch_all_attrs()

    bbox_list: Optional[List[float]] = None
    if bbox:
        try:
            parts = [float(p) for p in bbox.split(",")]
            if len(parts) == 4:
                bbox_list = parts
        except Exception:
            bbox_list = None

    fc = _to_jsonable(join_geo(attrs, state_fips=state_fips, bbox=bbox_list))

    if not state_fips and not bbox:
        _full_cache.update({"ts": now, "fc": fc})

    return Response(
        content=json.dumps(fc),
        media_type="application/geo+json"
    )

@app.get("/risk/counties/{state_fips}.geojson")
async def counties_by_state(state_fips: str):
    attrs = await fetch_all_attrs()
    fc = _to_jsonable(join_geo(attrs, state_fips=state_fips))
    return Response(content=json.dumps(fc), media_type="application/geo+json")

@app.get("/risk/county/{fips}")
async def county_detail(fips: str):
    fips = str(fips).zfill(5)
    row = await fetch_one_county(fips)
    if not row:
        raise HTTPException(status_code=404, detail="County not found")

    # compact hazard_ratings map for the popup (hazard -> rating)
    hazard_ratings: Dict[str, str] = {}
    hazards = row.get("hazards") or {}
    if isinstance(hazards, dict):
        for name in HAZARD_ORDER:
            v = hazards.get(name) or {}
            rating = v.get("rating") or v.get("risk_rating")
            if rating:
                hazard_ratings[name] = rating

    payload = {
        "fips": row["fips"],
        "county_name": row["county_name"],
        "state_name": row["state_name"],
        "state_fips": row["state_fips"],
        "nri_composite_rating": row["nri_composite_rating"],
        "nri_composite_score": row["nri_composite_score"],
        "hazard_ratings": hazard_ratings,
        "hazards": hazards,
    }
    return JSONResponse(_to_jsonable(payload))


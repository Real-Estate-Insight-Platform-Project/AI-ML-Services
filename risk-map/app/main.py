# risk-map/app/main.py
import os, time, json
from decimal import Decimal
from typing import Any, Dict, Optional, List, Tuple

import orjson
import asyncpg
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response, JSONResponse
from dotenv import load_dotenv

load_dotenv()

DB_URL    = os.environ["DATABASE_URL"]
GEO_PATH  = os.environ.get("COUNTIES_GEOJSON", "data/counties.min.geojson")
CACHE_TTL = int(os.getenv("NRI_GEOJSON_TTL", "86400"))  # seconds

# ---------------------------------------------------------------------
# App + middleware (gzip large GeoJSON payloads)  [FastAPI/Starlette]
# ---------------------------------------------------------------------
app = FastAPI(title="Risk Map API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
app.add_middleware(GZipMiddleware, minimum_size=1500)  # compress larger replies. :contentReference[oaicite:1]{index=1}


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def _parse_bbox(bbox: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not bbox:
        return None
    try:
        a, b, c, d = [float(x) for x in bbox.split(",")]
        return (a, b, c, d)
    except Exception:
        return None


# ---------------- load polygons once + precompute simple bbox ----------------
with open(GEO_PATH, "r", encoding="utf-8") as f:
    _poly_fc = json.load(f)  # FeatureCollection with properties.GEOID/STATEFP/NAME

def _geom_bbox(geom: Dict[str, Any]) -> Tuple[float, float, float, float]:
    # compute lon/lat bbox of a (Multi)Polygon
    xs, ys = [], []
    t = geom["type"]
    if t == "Polygon":
        for ring in geom["coordinates"]:
            for x, y in ring:
                xs.append(x); ys.append(y)
    elif t == "MultiPolygon":
        for poly in geom["coordinates"]:
            for ring in poly:
                for x, y in ring:
                    xs.append(x); ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys))

# compact in-memory list: (geoid, statefp, name, geometry, bbox)
POLY: List[Dict[str, Any]] = []
for feat in _poly_fc.get("features", []):
    props = feat.get("properties") or {}
    geoid = str(props.get("GEOID") or props.get("geoid") or "").zfill(5)
    statefp = str(props.get("STATEFP") or "").zfill(2)
    if not geoid:
        continue
    POLY.append({
        "geoid": geoid,
        "statefp": statefp,
        "name": props.get("NAME"),
        "geometry": feat["geometry"],
        "bbox": _geom_bbox(feat["geometry"]),
    })


def _bbox_intersects(a: Tuple[float, float, float, float],
                     b: Tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


# ---------------- connection pool (PgBouncer-safe) ----------------
@app.on_event("startup")
async def _startup():
    app.state.pool = await asyncpg.create_pool(
        DB_URL,
        min_size=1, max_size=6,
        statement_cache_size=0,  # disable named prepared statements behind PgBouncer. :contentReference[oaicite:2]{index=2}
        max_inactive_connection_lifetime=60.0,
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
             risk_index_score, risk_index_rating, predominant_hazard
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
            "predominant_hazard": r["predominant_hazard"],
        }
    return out

async def fetch_one_county(fips: str) -> Optional[Dict[str, Any]]:
    sql = """
      select county_fips, county_name, state_name, state_fips,
             risk_index_score, risk_index_rating, predominant_hazard
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
        "predominant_hazard": row["predominant_hazard"],
    }


# ---------------- join polygons + attributes ----------------
def join_geo(attrs: Dict[str, Dict[str, Any]],
             state_fips: Optional[str] = None,
             bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
    features = []
    for p in POLY:
        if state_fips and p["statefp"] != str(state_fips).zfill(2):
            continue
        if bbox and not _bbox_intersects(p["bbox"], bbox):
            continue

        a = attrs.get(p["geoid"])
        props = {
            "county_fips": p["geoid"],
            "STATEFP": p["statefp"],
            "NAME": p["name"],
        }
        if a:
            props.update({
                "county_name": a.get("county_name") or p["name"],
                "state_name":  a.get("state_name"),
                "state_fips":  a.get("state_fips"),
                "risk_rating_composite": a.get("risk_index_rating"),
                "risk_score_composite":  a.get("risk_index_score"),
                "predominant_hazard":    a.get("predominant_hazard"),
            })

        features.append({"type": "Feature", "geometry": p["geometry"], "properties": props})

    return {"type": "FeatureCollection", "features": features}


# simple in-memory cache for full-US response
_full_cache: Dict[str, Any] = {"ts": 0, "fc": None}


# ---------------- routes ----------------
@app.get("/", tags=["health"])
async def root():
    return {"ok": True, "service": "risk-map", "polygons": len(POLY)}

@app.get("/risk/counties.geojson")
async def counties_geojson(
    state_fips: Optional[str] = Query(default=None, description="2-digit state FIPS"),
    bbox: Optional[str]     = Query(default=None, description="minx,miny,maxx,maxy (lon/lat)")
):
    # return cached nationwide layer if no filters and cache live
    now = time.time()
    if not state_fips and not bbox and _full_cache["fc"] and now - _full_cache["ts"] < CACHE_TTL:
        return Response(content=orjson.dumps(_full_cache["fc"]),
                        media_type="application/geo+json")  # correct GeoJSON media type. :contentReference[oaicite:3]{index=3}

    attrs = await fetch_all_attrs()
    fc = join_geo(attrs, state_fips=state_fips, bbox=_parse_bbox(bbox))
    fc = _to_jsonable(fc)

    if not state_fips and not bbox:
        _full_cache.update({"ts": now, "fc": fc})

    return Response(content=orjson.dumps(fc), media_type="application/geo+json")

@app.get("/risk/counties/{state_fips}.geojson")
async def counties_by_state(state_fips: str):
    attrs = await fetch_all_attrs()
    fc = _to_jsonable(join_geo(attrs, state_fips=state_fips))
    return Response(content=orjson.dumps(fc), media_type="application/geo+json")

@app.get("/risk/county/{fips}")
async def county_detail(fips: str):
    fips = str(fips).zfill(5)
    row = await fetch_one_county(fips)
    if not row:
        raise HTTPException(status_code=404, detail="County not found")

    payload = {
        "fips": row["fips"],
        "county_name": row["county_name"],
        "state_name": row["state_name"],
        "state_fips": row["state_fips"],
        "nri_composite_rating": row["nri_composite_rating"],
        "nri_composite_score": row["nri_composite_score"],
        "predominant_hazard": row["predominant_hazard"],  # for popup
    }
    return JSONResponse(_to_jsonable(payload))

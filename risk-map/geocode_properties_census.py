import os, time, requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL")

CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

S = requests.Session()
S.headers.update({"User-Agent": f"RealEstate-Geocoder/1.0 ({CONTACT_EMAIL})"})

def one_line(addr, city, state, zip_code):
    parts = [str(addr or "").strip(), str(city or "").strip(), str(state or "").strip(), str(zip_code or "").strip()]
    return ", ".join([p for p in parts if p])

def geocode_census(oneline: str, pause=0.2):
    params = {"address": oneline, "benchmark": "Public_AR_Current", "format": "json"}
    r = S.get(CENSUS_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    matches = data.get("result", {}).get("addressMatches", [])
    if not matches:
        time.sleep(pause)  # polite pacing even on miss
        return None
    coords = matches[0].get("coordinates") or {}
    lon, lat = coords.get("x"), coords.get("y")
    time.sleep(pause)
    if lon is None or lat is None:
        return None
    return float(lat), float(lon)

def fetch_batch(limit=200, offset=0):
    """
    Pull rows that still have NULL location, deterministically ordered.
    Using offset pagination so each pass can walk forward through the set.
    """
    resp = (
        sb.table("properties")
        .select("id,address,city,state,zip_code")
        .is_("location", "null")
        .order("id")            # deterministic
        .range(offset, offset + limit - 1)
        .execute()
    )
    return resp.data or []

def set_location(row_id, lat, lon):
    sb.rpc("set_property_location", {"p_id": row_id, "p_lat": lat, "p_lon": lon}).execute()

def main():
    BATCH = 200
    total_updated = 0
    seen_ids = set()

    # “No progress breaker”: if we complete a full sweep with no updates, stop.
    # We implement a sweep as: keep paginating until the server returns <BATCH rows.
    while True:
        offset = 0
        progress_this_sweep = False
        processed_any = False

        while True:
            rows = fetch_batch(limit=BATCH, offset=offset)
            if not rows:
                break  # no rows at this offset

            processed_any = True
            updated_in_chunk = 0

            for r in rows:
                row_id = r["id"]
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)

                addr_str = one_line(r.get("address"), r.get("city"), r.get("state"), r.get("zip_code"))
                try:
                    result = geocode_census(addr_str)
                except requests.RequestException:
                    result = None

                if result:
                    lat, lon = result
                    set_location(row_id, lat, lon)
                    total_updated += 1
                    updated_in_chunk += 1
                    print(f"✔ {row_id}  {lat:.6f},{lon:.6f}  <- {addr_str}")
                else:
                    print(f"✖ {row_id}  no match  <- {addr_str}")

            # If nothing updated in this chunk, we still advance offset to avoid re-reading it next loop.
            offset += BATCH

            if updated_in_chunk > 0:
                progress_this_sweep = True

            # If server gave fewer than BATCH rows, we reached the end of the current NULL set.
            if len(rows) < BATCH:
                break

        # If there were no rows at all, we're done.
        if not processed_any:
            break

        # If we processed a full sweep but updated nothing, remaining rows look like “permanent” misses.
        # Break to avoid looping forever.
        if not progress_this_sweep:
            print("No progress in this sweep (all remaining rows look like no-matches). Stopping.")
            break

        # Otherwise, loop again for another sweep; some rows may have been added/changed concurrently.

    print(f"Done. Updated {total_updated} rows.")

if __name__ == "__main__":
    main()

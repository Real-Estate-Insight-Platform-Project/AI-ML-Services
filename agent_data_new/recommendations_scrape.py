# recommendations_scrape.py
import os, json, time
import pandas as pd
import requests
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_RECOMMENDATION")  
if not RAPIDAPI_KEY:
    raise SystemExit("RAPIDAPI_RECOMMENDATION is not set. Add it to .env or environment before running.")

# --- API config ---
HOST = "us-realtor.p.rapidapi.com"
ENDPOINT = f"https://{HOST}/api/v1/agents/recommendation"

# --- pacing / retry config ---
BASE_SLEEP = 1.0
MAX_RETRIES = 4
BACKOFF = 1.7
TIMEOUT_S = 30

# --- session setup ---
session = requests.Session()
session.headers.update({
    "x-rapidapi-host": HOST,
    "x-rapidapi-key": RAPIDAPI_KEY,
})

def fetch_recommendations(advertiser_id: str):
    """Fetch recommendation payload for a single advertiser_id."""
    params = {"advertiserId": str(advertiser_id).strip()}
    delay = BASE_SLEEP
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(ENDPOINT, params=params, timeout=TIMEOUT_S)
            status = r.status_code
            if status == 200:
                try:
                    return {"advertiser_id": advertiser_id, "status": status, "payload": r.json(), "error": None}
                except ValueError:
                    return {"advertiser_id": advertiser_id, "status": status, "payload": None, "error": "non_json_response"}

            if status in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= BACKOFF
                continue

            return {"advertiser_id": advertiser_id, "status": status, "payload": None, "error": r.text[:1000]}
        except requests.RequestException:
            time.sleep(delay)
            delay *= BACKOFF

    return {"advertiser_id": advertiser_id, "status": None, "payload": None, "error": "max_retries_exceeded"}

def main(
    ids_csv_path: str = "review_zero_recommendation_nonzero_advertiser_ids.csv",  # ✅ input CSV
    out_json_path: str = "agent_recommendations.json",  # ✅ output JSON
    limit: int | None = None
):
    # Load IDs
    df_ids = pd.read_csv(ids_csv_path)
    if "advertiser_id" not in df_ids.columns:
        raise SystemExit("❌ Expected column 'advertiser_id' in input CSV.")

    ids = (
        df_ids["advertiser_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )

    if limit:
        ids = ids[:limit]

    print(f"🟡 Starting recommendation scrape for {len(ids)} advertiser IDs...")

    out = []
    for i, aid in enumerate(ids, 1):
        rec = fetch_recommendations(aid)
        out.append(rec)
        time.sleep(BASE_SLEEP)

        if i % 25 == 0 or i == len(ids):
            print(f"[{i}/{len(ids)}] fetched")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(out)} recommendation payloads → {out_json_path}")

if __name__ == "__main__":
    # Optional: main(limit=50) for test
    main()

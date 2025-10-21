# convert_recommendations_to_csv.py
import json
import pandas as pd

INPUT_JSON = "agent_recommendations.json"
OUTPUT_CSV  = "agent_recommendations_flat.csv"

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def explode_recommendations(records) -> list[dict]:
    rows = []
    for rec in records or []:
        advertiser_id = (rec or {}).get("advertiser_id")
        payload = (rec or {}).get("payload") or {}
        data = payload.get("data") or []

        for r in data:
            rows.append({
                "advertiser_id": advertiser_id,
                "recommendation_id": r.get("id"),
                "comment": r.get("comment"),
                "display_name": r.get("display_name"),
                "email": r.get("email"),
                "relation": r.get("relation"),        # BUYER/SELLER/etc.
                "year": r.get("year"),
                "started_timestamp": r.get("started_timestamp"),
                "last_updated": r.get("last_updated"),
                "location": r.get("location"),
                "address": r.get("address"),
                "source_id": r.get("source_id"),
                "photo": r.get("photo"),
                "video": r.get("video"),
            })
    return rows

def main():
    records = load_json(INPUT_JSON)
    rows = explode_recommendations(records)
    if not rows:
        print("No recommendations found. Nothing written.")
        return

    df = pd.DataFrame(rows)

    # nice column order
    preferred = [
        "advertiser_id","recommendation_id","relation","comment",
        "display_name","email","location","address",
        "year","started_timestamp","last_updated",
        "source_id","photo","video"
    ]
    df = df[[c for c in preferred if c in df.columns] +
            [c for c in df.columns if c not in preferred]]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ Wrote {len(df)} recommendation rows → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

# convert_reviews_to_csv.py
import json
import pandas as pd

INPUT_JSON = "agent_reviews.json"
OUTPUT_CSV  = "agent_reviews_flat.csv"

def load_reviews(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def explode_reviews(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for rec in records or []:
        advertiser_id = (rec or {}).get("advertiser_id")
        payload = (rec or {}).get("payload") or {}
        data = payload.get("data") or []

        for rv in data:
            rows.append({
                "advertiser_id": advertiser_id,
                "review_id": rv.get("id"),
                "rating": rv.get("rating"),
                "comment": rv.get("comment"),
                "display_name": rv.get("display_name"),
                "describe_yourself": rv.get("describe_yourself"),
                "location": rv.get("location"),
                "year": rv.get("year"),
                "transaction_date": rv.get("transaction_date"),
                "started_timestamp": rv.get("started_timestamp"),
                "source_id": rv.get("source_id"),
                "responsiveness": rv.get("responsiveness"),
                "negotiation_skills": rv.get("negotiation_skills"),
                "professionalism_communication": rv.get("professionalism_communication"),
                "market_expertise": rv.get("market_expertise"),
                "link": rv.get("link"),
                "reply": rv.get("reply"),
            })
    return rows

def main():
    records = load_reviews(INPUT_JSON)
    rows = explode_reviews(records)
    if not rows:
        print("No reviews found in JSON. Nothing written.")
        return

    df = pd.DataFrame(rows)
    preferred = [
        "advertiser_id","review_id","rating","comment","display_name",
        "describe_yourself","location","year","transaction_date",
        "started_timestamp","source_id","responsiveness","negotiation_skills",
        "professionalism_communication","market_expertise","link","reply"
    ]
    df = df[[c for c in preferred if c in df.columns] +
            [c for c in df.columns if c not in preferred]]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ Wrote {len(df)} review rows → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

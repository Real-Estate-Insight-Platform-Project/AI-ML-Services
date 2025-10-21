# json_to_csv_reviews.py
import os
import json
from datetime import datetime
from typing import Any, List, Optional
import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
INPUT_JSON_PATH = os.path.join(BASE_DIR, "scraped_data", "agent_reviews.json")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "converted", "reviews.csv")

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def smart_title_loose(text: Optional[str]) -> str:
    if not text:
        return ""
    words = " ".join(str(text).split()).split(" ")
    return " ".join(w[:1].upper() + w[1:].lower() if w else "" for w in words)

def normalize_location(loc: Optional[str]) -> str:
    """'ANCHORAGE, ak' -> 'Anchorage, AK' | Handles hyphenated city parts."""
    if not loc:
        return ""
    loc = " ".join(str(loc).replace("-", " - ").split())
    if "," in loc:
        city, state = [p.strip() for p in loc.split(",", 1)]
        city = smart_title_loose(city).replace(" - ", "-")
        state = state.upper()
        return f"{city}, {state}" if len(state) == 2 else f"{city}, {smart_title_loose(state)}"
    return smart_title_loose(loc)

def normalize_comment(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(" ".join(line.split()) for line in s.split("\n")).strip()

def parse_date_yyyy_mm_dd(date_str: Optional[str]) -> str:
    """Return YYYY-MM-DD or ''."""
    if not date_str:
        return ""
    try:
        ds = str(date_str)
        if ds.endswith("Z"):
            ds = ds.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ds)
        return dt.date().isoformat()
    except Exception:
        dt = pd.to_datetime(str(date_str), utc=True, errors="coerce")
        return "" if pd.isna(dt) else dt.date().isoformat()

def clamp_rating(val: Any, lo: int = 1, hi: int = 5) -> Optional[int]:
    """Keep 1..5 else None."""
    try:
        f = float(val)
    except Exception:
        return None
    i = int(round(f))
    return i if (lo <= i <= hi) else None

def clamp_subscore(val: Any) -> Optional[int]:
    """Keep None for invalid; keep 0 if present; otherwise 1..5."""
    try:
        f = float(val)
    except Exception:
        return None
    i = int(round(f))
    if i == 0:
        return 0
    return i if (1 <= i <= 5) else None

# ------------------------------------------------------------
# CORE
# ------------------------------------------------------------
def load_reviews(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def explode_reviews(records: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for rec in records or []:
        advertiser_id = (rec or {}).get("advertiser_id")
        payload = (rec or {}).get("payload") or {}
        data = payload.get("data") or []

        for rv in data:
            rows.append({
                # Identity
                "advertiser_id": str(advertiser_id) if advertiser_id is not None else None,
                "review_id": rv.get("id"),

                # Core review
                "review_rating": clamp_rating(rv.get("rating")),
                "review_comment": normalize_comment(rv.get("comment")),
                "review_created_date": parse_date_yyyy_mm_dd(rv.get("started_timestamp")),
                "transaction_date": parse_date_yyyy_mm_dd(rv.get("transaction_date")),

                # Reviewer — keep role AS-IS (only strip)
                "reviewer_role": (rv.get("describe_yourself") or "").strip(),
                "reviewer_location": normalize_location(rv.get("location")),

                # Subscores
                "sub_responsiveness": clamp_subscore(rv.get("responsiveness")),
                "sub_negotiation": clamp_subscore(rv.get("negotiation_skills")),
                "sub_professionalism": clamp_subscore(rv.get("professionalism_communication")),
                "sub_market_expertise": clamp_subscore(rv.get("market_expertise")),
            })
    return rows

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)
    df = df.copy()  # <-- ensure we're working on our own frame

    # Keep if has a rating or a non-empty comment
    df["review_comment"] = df["review_comment"].fillna("").astype(str)
    df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce")

    df = df.loc[(df["review_rating"].notna()) | (df["review_comment"].str.len() > 0)].copy()

    # Coerce subscores (use .loc to avoid chained assignment warning)
    for c in ["sub_responsiveness", "sub_negotiation", "sub_professionalism", "sub_market_expertise"]:
        if c in df.columns:
            df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")

    # Deduplicate (advertiser_id, review_id)
    if {"advertiser_id", "review_id"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["advertiser_id", "review_id"], keep="first")

    # Sort by advertiser then newest review date, then transaction_date
    df = df.sort_values(
        by=["advertiser_id", "review_created_date", "transaction_date"],
        ascending=[True, False, False],
        na_position="last"
    )

    print(f"  ✓ Final valid reviews: {len(df)} (removed {initial - len(df)})")
    return df

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        "advertiser_id", "review_id",
        "review_rating", "review_comment",
        "review_created_date", "transaction_date",
        "reviewer_role", "reviewer_location",
        "sub_responsiveness", "sub_negotiation", "sub_professionalism", "sub_market_expertise",
    ]
    extra = [c for c in df.columns if c not in ordered]
    return df[ordered + extra] if extra else df[ordered]

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def json_to_csv_reviews(input_path: str, output_path: str) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("REVIEWS JSON → CSV")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}\n")

    if not os.path.exists(input_path):
        print(f"✗ ERROR: Input not found: {input_path}")
        return pd.DataFrame()

    records = load_reviews(input_path)
    rows = explode_reviews(records)
    if not rows:
        print("✗ ERROR: No reviews found in JSON.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print("Validating and cleaning...")
    df = validate_and_clean(df)
    df = reorder_columns(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print("\n" + "=" * 70)
    print("✓ CONVERSION SUCCESSFUL")
    print("=" * 70)
    print(f"Output file: {output_path}")
    print(f"Total reviews: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("=" * 70)
    return df

if __name__ == "__main__":
    json_to_csv_reviews(INPUT_JSON_PATH, OUTPUT_CSV_PATH)

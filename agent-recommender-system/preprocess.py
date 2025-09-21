# preprocess.py
import pandas as pd
import numpy as np
import ast, json, re, math
from typing import List, Any

# ---------- list parsing & cleaning ----------
def _parse_jsonish_lists(x: Any):
    if isinstance(x, list) or pd.isna(x):
        return [] if pd.isna(x) else x
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "[]"}:
        return []
    try:
        return ast.literal_eval(s) if s.startswith("[") else [s]
    except Exception:
        try:
            s2 = re.sub(r",\s*([}\]])", r"\1", s.replace("'", '"'))
            v = json.loads(s2)
            return v if isinstance(v, list) else [v]
        except Exception:
            return [s]

def _ensure_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(i).strip() for i in v if str(i).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []

def _clean_prices(seq: Any) -> List[float]:
    if not isinstance(seq, list):
        return []
    out = []
    for val in seq:
        try:
            f = float(val)
            if math.isfinite(f) and 1_000 <= f <= 50_000_000:
                out.append(f)
        except Exception:
            pass
    return out

# ---------- convert Python list -> Postgres text[] literal ----------
def _to_pg_text_array(xs: List[str]) -> str:
    """Return a Postgres array literal like {"a","b"} suitable for CSV import to text[]."""
    if not isinstance(xs, list) or not xs:
        return "{}"
    # escape embedded double quotes
    items = ['"{}"'.format(str(x).replace('"', r'\"')) for x in xs]
    return "{" + ",".join(items) + "}"

def preprocess_csv(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)

    # ---- Lists ----
    for col in ["primaryServiceRegions", "propertyTypes", "dealPrices", "languages"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_jsonish_lists).apply(_ensure_list)
        else:
            df[col] = [[]] * len(df)

    # ---- Numerics ----
    for col in ["starRating","numReviews","pastYearDeals","pastYearDealsInRegion",
                "homeTransactionsLifetime","transactionVolumeLifetime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Deal price statistics ----
    prices = df["dealPrices"].apply(_clean_prices)
    df["dealPrices_count"]  = prices.apply(len)
    df["dealPrices_median"] = prices.apply(lambda v: float(np.median(v)) if v else 0.0)
    df["dealPrices_q25"]    = prices.apply(lambda v: float(np.percentile(v,25)) if v else 0.0)
    df["dealPrices_q75"]    = prices.apply(lambda v: float(np.percentile(v,75)) if v else 0.0)
    df["dealPrices_min"]    = prices.apply(lambda v: float(np.min(v)) if v else 0.0)
    df["dealPrices_max"]    = prices.apply(lambda v: float(np.max(v)) if v else 0.0)
    df["dealPrices_std"]    = prices.apply(lambda v: float(np.std(v)) if len(v) > 1 else 0.0)
    df["price_range_span"]  = df["dealPrices_max"] - df["dealPrices_min"]
    df["price_coefficient_variation"] = np.where(
        df["dealPrices_median"] > 0,
        df["dealPrices_std"] / df["dealPrices_median"],
        0
    )

    # ---- Defaults ----
    df["starRating"] = df["starRating"].fillna(0.0)
    df["numReviews"] = df["numReviews"].fillna(0).astype(int)
    df["pastYearDeals"] = df["pastYearDeals"].fillna(0).astype(int)
    for c, default in [("officeState","Unknown"),
                       ("brokerageName","Independent"),
                       ("phoneNumber","Not Available")]:
        if c in df:
            df[c] = df[c].fillna(default)

    # ---- Impute lifetime counts/volume from price history ----
    df["homeTransactionsLifetime"] = df["homeTransactionsLifetime"].fillna(df["dealPrices_count"])
    calc_volume = prices.apply(lambda v: float(sum(v)) if v else 0.0)
    df["transactionVolumeLifetime"] = df["transactionVolumeLifetime"].fillna(calc_volume)

    # ---- Derived metrics ----
    safe_den = df["homeTransactionsLifetime"].replace(0, np.nan)
    df["avg_transaction_value"] = (df["transactionVolumeLifetime"] / safe_den).fillna(0)
    df["experience_score"] = np.log1p(df["homeTransactionsLifetime"])
    df["recent_activity_ratio"] = np.where(
        df["homeTransactionsLifetime"] > 0,
        df["pastYearDeals"] / df["homeTransactionsLifetime"], 0
    )
    df["weighted_rating"] = df["starRating"] * np.log1p(df["numReviews"])
    df["num_service_regions"] = df["primaryServiceRegions"].apply(len)
    df["num_property_types"]  = df["propertyTypes"].apply(len)
    df["market_breadth_score"] = df["num_service_regions"] * df["num_property_types"]
    df["specialization_index"] = np.where(df["num_property_types"]>0, 1/df["num_property_types"], 0)
    df["price_tier_percentile"] = (
        df["dealPrices_median"].rank(pct=True) if df["dealPrices_median"].max() > 0 else 0.5
    )

    # ---- Booleans ----
    for b in ["partner","isPremier","servesOffers","servesListings","isActive","profileContactEnabled"]:
        if b in df.columns:
            df[b] = df[b].astype(bool)
        else:
            df[b] = False

    # ---- Rename to snake_case for Postgres ----
    rename = {
        "agentId":"agent_id","brokerageName":"brokerage_name","officeState":"office_state",
        "phoneNumber":"phone_number","profileUrl":"profile_url","photoUrl":"photo_url",
        "starRating":"star_rating","numReviews":"num_reviews","pastYearDeals":"past_year_deals",
        "pastYearDealsInRegion":"past_year_deals_in_region",
        "homeTransactionsLifetime":"home_transactions_lifetime",
        "transactionVolumeLifetime":"transaction_volume_lifetime",
        "primaryServiceRegions":"primary_service_regions","propertyTypes":"property_types",
        "dealPrices_count":"deal_prices_count","dealPrices_median":"deal_prices_median",
        "dealPrices_q25":"deal_prices_q25","dealPrices_q75":"deal_prices_q75",
        "dealPrices_min":"deal_prices_min","dealPrices_max":"deal_prices_max",
        "dealPrices_std":"deal_prices_std",
        "price_range_span":"price_range_span","price_coefficient_variation":"price_coefficient_variation",
        "price_tier_percentile":"price_tier_percentile",
        "market_breadth_score":"market_breadth_score","specialization_index":"specialization_index",
        "experience_score":"experience_score","recent_activity_ratio":"recent_activity_ratio",
        "weighted_rating":"weighted_rating",
        "isPremier":"is_premier","servesOffers":"serves_offers","servesListings":"serves_listings",
        "isActive":"is_active","profileContactEnabled":"profile_contact_enabled"
    }
    df = df.rename(columns=rename)

    # ── DEDUPE by agent_id (keep the strongest record) ──
    if "agent_id" in df.columns:
        df["agent_id"] = pd.to_numeric(df["agent_id"], errors="coerce")
        for c in ["past_year_deals","num_reviews","star_rating",
                  "transaction_volume_lifetime","deal_prices_count"]:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        before = len(df)
        df = (
            df.sort_values(
                ["agent_id","past_year_deals","num_reviews","star_rating",
                 "transaction_volume_lifetime","deal_prices_count"],
                ascending=[True, False, False, False, False, False]
            )
            .drop_duplicates(subset="agent_id", keep="first")
        )
        removed = before - len(df)
        if removed > 0:
            print(f"🧹 Removed {removed} duplicate rows by agent_id")
    # (pandas drop_duplicates API referenced here.) :contentReference[oaicite:1]{index=1}

    # ---- Keep only columns expected by DB/API ----
    keep = [
        'agent_id','slug','name','brokerage_name','office_state','email','phone_number',
        'profile_url','photo_url','star_rating','num_reviews','past_year_deals',
        'past_year_deals_in_region','home_transactions_lifetime','transaction_volume_lifetime',
        'avg_transaction_value','primary_service_regions','property_types','num_service_regions',
        'num_property_types','deal_prices_count','deal_prices_median','deal_prices_q25',
        'deal_prices_q75','deal_prices_min','deal_prices_max','deal_prices_std',
        'price_range_span','price_coefficient_variation','price_tier_percentile',
        'market_breadth_score','specialization_index','experience_score',
        'recent_activity_ratio','weighted_rating','partner','is_premier','serves_offers',
        'serves_listings','is_active','profile_contact_enabled'
    ]
    df = df[keep]

    # ---- Convert list columns to Postgres text[] literal for CSV import ----
    for col in ["primary_service_regions", "property_types"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_pg_text_array)  # produces {"a","b"} format
    # (Postgres array literal uses curly braces; this is the expected CSV input for text[].) :contentReference[oaicite:2]{index=2}

    df.to_csv(output_csv, index=False)
    print(f"✅ Wrote clean CSV to: {output_csv}")

if __name__ == "__main__":
    preprocess_csv("all_agents_combined.csv", "df_clean.csv")

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

# ---------- standardize geographical data ----------
def _normalize_business_market(market_name: str) -> str:
    """Normalize business market names to standard 'City, State' format."""
    if pd.isna(market_name) or not str(market_name).strip():
        return "Unknown"
    
    normalized = str(market_name).strip()
    
    # Common standardizations
    normalized = re.sub(r'\s+', ' ', normalized)  # normalize whitespace
    
    # Handle common city abbreviations and variations
    replacements = {
        'Ft ': 'Fort ', 'Ft. ': 'Fort ',
        'St ': 'Saint ', 'St. ': 'Saint ',
        'Mt ': 'Mount ', 'Mt. ': 'Mount ',
        ' Nyc': ' New York City', 'NYC': 'New York City',
        ' La ': ' Los Angeles ', 'LA': 'Los Angeles',
        ' Sf': ' San Francisco', 'SF': 'San Francisco',
        ' Dc': ' Washington', 'DC': 'Washington, D.C.'
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    # If it's already in "City, State" format, return as title case
    if ', ' in normalized:
        city, state = normalized.split(', ', 1)
        return f"{city.title()}, {state.title()}"
    
    # If it's just a city name, try to infer state (for known cities)
    city_to_state = {
        'Miami': 'Florida', 'Tampa': 'Florida', 'Orlando': 'Florida', 'Jacksonville': 'Florida',
        'Fort Myers': 'Florida', 'Naples': 'Florida', 'Cape Coral': 'Florida',
        'New York': 'New York', 'Los Angeles': 'California', 'Chicago': 'Illinois',
        'Houston': 'Texas', 'Phoenix': 'Arizona', 'Philadelphia': 'Pennsylvania',
        'San Antonio': 'Texas', 'San Diego': 'California', 'Dallas': 'Texas',
        'San Jose': 'California', 'Austin': 'Texas', 'Fort Worth': 'Texas',
        'Columbus': 'Ohio', 'Charlotte': 'North Carolina', 'San Francisco': 'California',
        'Indianapolis': 'Indiana', 'Seattle': 'Washington', 'Denver': 'Colorado',
        'Washington': 'D.C.', 'Boston': 'Massachusetts', 'Nashville': 'Tennessee',
        'Oklahoma City': 'Oklahoma', 'Las Vegas': 'Nevada', 'Portland': 'Oregon',
        'Memphis': 'Tennessee', 'Louisville': 'Kentucky', 'Milwaukee': 'Wisconsin',
        'Albuquerque': 'New Mexico', 'Tucson': 'Arizona', 'Fresno': 'California',
        'Sacramento': 'California', 'Kansas City': 'Missouri', 'Mesa': 'Arizona',
        'Atlanta': 'Georgia', 'Omaha': 'Nebraska', 'Colorado Springs': 'Colorado',
        'Raleigh': 'North Carolina', 'Virginia Beach': 'Virginia', 'Long Beach': 'California',
        'Miami Beach': 'Florida', 'Clearwater': 'Florida', 'St Petersburg': 'Florida'
    }
    
    normalized_title = normalized.title()
    if normalized_title in city_to_state:
        return f"{normalized_title}, {city_to_state[normalized_title]}"
    
    # If no state mapping found, return as title case
    return normalized_title

def _combine_service_areas(primary_regions: List[str], business_market: str, office_state: str) -> List[str]:
    """Combine different geographical indicators into a comprehensive service area list with City, State format."""
    service_areas = set()
    
    # Add primary service regions (convert to City, State format where possible)
    if isinstance(primary_regions, list):
        for region in primary_regions:
            if region and str(region).strip():
                normalized_region = _normalize_business_market(str(region).strip())
                service_areas.add(normalized_region)
    
    # Add business market
    if business_market and str(business_market).strip() not in ["Unknown", "nan", "None"]:
        normalized_market = _normalize_business_market(business_market)
        if normalized_market != "Unknown":
            service_areas.add(normalized_market)
    
    # Add state-level coverage
    if office_state and str(office_state).strip() not in ["Unknown", "nan", "None"]:
        state = str(office_state).strip()
        # Convert state abbreviations to full names
        state_abbrev_to_name = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
        }
        
        full_state = state_abbrev_to_name.get(state.upper(), state.title())
        service_areas.add(full_state)
    
    return sorted(list(service_areas))

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
                "homeTransactionsLifetime","transactionVolumeLifetime", "businessMarketId"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Geographical data standardization ----
    # Fill missing values first
    df["businessMarket"] = df["businessMarket"].fillna("Unknown")
    df["officeState"] = df["officeState"].fillna("Unknown")
    
    # Normalize business market names
    df["business_market_normalized"] = df["businessMarket"].apply(_normalize_business_market)
    
    # Create comprehensive service areas combining all geographical data
    df["comprehensive_service_areas"] = df.apply(
        lambda row: _combine_service_areas(
            row["primaryServiceRegions"],
            row["businessMarket"],
            row["officeState"]
        ),
        axis=1
    )

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
    for c, default in [("brokerageName","Independent"),
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
    df["num_comprehensive_areas"] = df["comprehensive_service_areas"].apply(len)
    df["market_breadth_score"] = df["num_comprehensive_areas"] * df["num_property_types"]
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
        "businessMarket":"business_market","businessMarketId":"business_market_id",
        "business_market_normalized":"business_market_normalized",
        "comprehensive_service_areas":"comprehensive_service_areas",
        "dealPrices_count":"deal_prices_count","dealPrices_median":"deal_prices_median",
        "dealPrices_q25":"deal_prices_q25","dealPrices_q75":"deal_prices_q75",
        "dealPrices_min":"deal_prices_min","dealPrices_max":"deal_prices_max",
        "dealPrices_std":"deal_prices_std",
        "price_range_span":"price_range_span","price_coefficient_variation":"price_coefficient_variation",
        "price_tier_percentile":"price_tier_percentile",
        "market_breadth_score":"market_breadth_score","specialization_index":"specialization_index",
        "experience_score":"experience_score","recent_activity_ratio":"recent_activity_ratio",
        "weighted_rating":"weighted_rating","num_comprehensive_areas":"num_comprehensive_areas",
        "isPremier":"is_premier","servesOffers":"serves_offers","servesListings":"serves_listings",
        "isActive":"is_active","profileContactEnabled":"profile_contact_enabled"
    }
    df = df.rename(columns=rename)

    # ──  DEDUPE by agent_id (keep the strongest record) ──
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

    # ---- Keep only columns expected by DB/API ----
    keep = [
        'agent_id','slug','name','brokerage_name','office_state','email','phone_number',
        'profile_url','photo_url','star_rating','num_reviews','past_year_deals',
        'past_year_deals_in_region','home_transactions_lifetime','transaction_volume_lifetime',
        'avg_transaction_value','primary_service_regions','property_types',
        'business_market','business_market_id','business_market_normalized',
        'comprehensive_service_areas','num_service_regions','num_property_types',
        'num_comprehensive_areas','deal_prices_count','deal_prices_median','deal_prices_q25',
        'deal_prices_q75','deal_prices_min','deal_prices_max','deal_prices_std',
        'price_range_span','price_coefficient_variation','price_tier_percentile',
        'market_breadth_score','specialization_index','experience_score',
        'recent_activity_ratio','weighted_rating','partner','is_premier','serves_offers',
        'serves_listings','is_active','profile_contact_enabled'
    ]
    df = df[keep]

    # ---- Convert list columns to Postgres text[] literal for CSV import ----
    for col in ["primary_service_regions", "property_types", "comprehensive_service_areas"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_pg_text_array)  # produces {"a","b"} format

    df.to_csv(output_csv, index=False)
    print(f"✅ Wrote clean CSV to: {output_csv}")

if __name__ == "__main__":
    preprocess_csv("all_agents_combined.csv", "df_clean.csv")
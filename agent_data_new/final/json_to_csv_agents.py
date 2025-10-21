import os, json, re
from datetime import datetime, timezone
import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
INPUT_JSON_PATH = os.path.join(BASE_DIR, "scraped_data", "agents_by_state.json")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "converted", "agents.csv")

# ------------------------------------------------------------
# HELPERS (unchanged + new prettifiers + year/month validation)
# ------------------------------------------------------------
def consolidate_list(values):
    """Accepts list/str/None; returns comma-separated string without empties."""
    if values is None:
        return ""
    if isinstance(values, str):
        return values.strip()
    if isinstance(values, (list, tuple, set)):
        items = [str(v).strip() for v in values if v not in (None, "", [], {})]
        return ", ".join(items)
    return str(values)

def smart_title(name):
    """Title-case names incl. hyphens/apostrophes (e.g., O'Neil, Jean-Luc)."""
    if not name:
        return ""
    parts = []
    for token in str(name).strip().split():
        token = "-".join(p.capitalize() for p in token.split("-") if p)
        token = "'".join(p.capitalize() for p in token.split("'"))
        parts.append(token)
    return " ".join(parts)

def smart_title_loose(text):
    """Simple title case for city names like 'ANCHORAGE' -> 'Anchorage'."""
    if not text:
        return ""
    return " ".join(w.capitalize() for w in str(text).split())

def normalize_us_phone(number, ext=None):
    """Format as (AAA) BBB-CCCC, preserve ext (x123). Tolerant to messy input."""
    if not number:
        return ""
    digits = "".join(ch for ch in str(number) if ch.isdigit())
    # Strip US country code if present
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        out = f"({digits[0:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 7:
        out = f"{digits[0:3]}-{digits[3:]}"
    else:
        # Fallback to original if not 7/10/11 digits
        out = str(number)
    ext_digits = "".join(ch for ch in str(ext) if ch.isdigit()) if ext else ""
    if ext_digits:
        out += f" x{ext_digits}"
    return out

def get_office_phone(office):
    """Best-effort extraction: office.phones[0].number -> office.phone_list.phone_1.number."""
    office = office or {}
    phones = office.get("phones") or []
    if phones and isinstance(phones[0], dict):
        return normalize_us_phone(phones[0].get("number"), phones[0].get("ext"))
    plist = office.get("phone_list") or {}
    p1 = plist.get("phone_1") or {}
    if isinstance(p1, dict) and p1.get("number"):
        return normalize_us_phone(p1.get("number"), p1.get("ext"))
    return ""

def normalize_url(url):
    """Ensure URLs have a scheme (https:// by default)."""
    if not url:
        return ""
    u = str(url).strip()
    if u.startswith(("http://", "https://")):
        return u
    return "https://" + u.lstrip("/")

def format_date(date_str):
    """To YYYY-MM-DD. Accepts ISO with/without Z; else returns ''."""
    if not date_str:
        return ""
    try:
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

def calculate_days_since(date_str):
    """Days from date_str (ISO/DATE) to now; returns None if invalid."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        return (datetime.now(dt.tzinfo) - dt).days
    except Exception:
        return None

# ---------- NEW: Office name/address prettifiers ----------
def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

ACRONYMS = {
    r"\bRE/?MAX\b": "RE/MAX",
    r"\bKW\b|\bKeller\s*Williams\b": "Keller Williams",
    r"\b(eXp|EXP)\s*Realty\b": "eXp Realty",
}

LEGAL_SUFFIXES = {
    r"\bL\.?L\.?C\.?\b": "LLC",
    r"\bINC\b|\bInc\b|\bIncorporated\b": "Inc.",
    r"\bCORP\b|\bCorp\b": "Corp.",
    r"\bCO\b(?!\w)": "Co.",
    r"\bLTD\b|\bLtd\b": "Ltd.",
    r"\bPLLC\b|\bP\.?L\.?L\.?C\.?\b": "PLLC",
    r"\bP\.?C\.?\b": "PC",
    r"\bRealtors\b": "REALTORS®",
    r"\bRealtor\b": "REALTOR®",
    r"\bRealty\b": "Realty",
    r"\bReal Estate\b": "Real Estate",
}

def prettify_office_name(name: str) -> str:
    if not name:
        return ""
    s = _clean_ws(name)
    # Titleish case per token unless it's all caps acronym
    s = " ".join(t if re.fullmatch(r"[A-Z]{2,}", t) else t.capitalize() for t in s.split())
    # Apply acronyms and legal suffix normalizations
    for pat, repl in ACRONYMS.items():
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    for pat, repl in LEGAL_SUFFIXES.items():
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return _clean_ws(s)

DIRECTIONALS = {"N","S","E","W","NE","NW","SE","SW"}
STREET_SUFFIXES = {"St","Ave","Blvd","Rd","Dr","Ln","Ct","Cir","Hwy","Pkwy","Ter"}

def _prettify_street_line(line: str) -> str:
    if not line:
        return ""
    line = _clean_ws(line)
    out = []
    for tok in line.split():
        raw = tok.strip(",")
        up = raw.upper()
        if re.fullmatch(r"\d+[A-Za-z]?", raw):
            out.append(raw)
        elif up in DIRECTIONALS:
            out.append(up)
        elif raw.capitalize() in STREET_SUFFIXES:
            out.append(raw.capitalize())
        else:
            t = "-".join(p.capitalize() for p in raw.split("-"))
            t = "'".join(p.capitalize() for p in t.split("'"))
            out.append(t)
    return " ".join(out)

def build_office_address(office_dict):
    """Prettified 'line, City, ST ZIP'."""
    addr = (office_dict or {}).get("address") or {}
    line = _prettify_street_line(addr.get("line"))
    city = smart_title_loose(addr.get("city"))
    state_code = (addr.get("state_code") or "").strip().upper()
    postal = (addr.get("postal_code") or "").strip()
    parts = [p for p in [line, city] if p]
    tail = " ".join(p for p in [state_code, postal] if p)
    if tail:
        parts.append(tail)
    return ", ".join(parts)

# ---------- NEW: first-year/month validation + experience ----------
def validate_first_active(year_raw, month_raw):
    """
    Year: 1960..current_year.
    Month: 1..12; treat 0/None as 1 if year valid.
    Clamp future month within the current year.
    Returns (year|None, month|None).
    """
    now = datetime.now(timezone.utc)
    this_y, this_m = now.year, now.month

    def _toi(v):
        try:
            return int(v)
        except:
            return None

    y = _toi(year_raw)
    m = _toi(month_raw)

    if y is None or y < 1960 or y > this_y:
        return (None, None)

    if m is None or m == 0:
        m = 1
    if m < 1 or m > 12:
        return (None, None)

    if y == this_y and m > this_m:
        m = this_m

    return (y, m)

def compute_experience_years(y, m):
    """Full years difference, floored; never negative. None if invalid."""
    if y is None or m is None:
        return None
    now = datetime.now(timezone.utc)
    years = now.year - y
    if now.month < m:
        years -= 1
    return max(int(years), 0)

def consolidate_designations(agent):
    desigs = set()
    for d in (agent.get("designations") or []):
        name = (d or {}).get("name")
        if name:
            desigs.add(str(name))
    for m in (agent.get("mls") or []):
        val = (m or {}).get("designation")
        if val:
            desigs.add(str(val))
    return consolidate_list(sorted(desigs))

def validate_and_clean(df):
    initial = len(df)
    # Deduplicate and enforce required IDs/names
    df = df.drop_duplicates(subset=["advertiser_id"], keep="first")
    df = df[df["advertiser_id"].notna()]
    df = df[df["full_name"].notna() & (df["full_name"] != "")]
    # Numeric coercions
    numeric_cols = [
        "active_listings_count","recently_sold_count","experience_years",
        "days_since_last_sale","review_count","agent_rating",
        "active_listings_min_price","active_listings_max_price",
        "recently_sold_min_price","recently_sold_max_price"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Booleans
    if "has_photo" in df.columns: df["has_photo"] = df["has_photo"].astype(bool)
    if "is_realtor" in df.columns: df["is_realtor"] = df["is_realtor"].astype(bool)
    print(f"  ✓ Final valid records: {len(df)} (removed {initial - len(df)})")
    return df

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def json_to_csv_agents(input_path, output_path):
    print("\n" + "="*70)
    print("AGENT JSON-TO-CSV CONVERSION")
    print("="*70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}\n")

    if not os.path.exists(input_path):
        print(f"✗ ERROR: Input not found: {input_path}")
        return None

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Allow single or multi-state file
    if isinstance(raw, dict):
        raw = [raw]

    agents_list, states_found = [], set()

    print("Processing agents...")
    for state_record in raw:
        state = state_record.get("state", "Unknown")
        usps  = state_record.get("usps", "XX")
        states_found.add(f"{state} ({usps})")

        payload = state_record.get("payload", {}) or {}
        data_block = payload.get("data", {}) or {}

        # Support both data.agents and data.pageData.agents
        if "agents" in data_block:
            agents = data_block.get("agents") or []
        else:
            agents = (data_block.get("pageData") or {}).get("agents") or []

        print(f"  Processing {state} ({usps}): {len(agents)} agents found")

        for agent in agents:
            # Skip absolutely inactive rows
            if not any([
                agent.get("agent_rating"),
                agent.get("review_count"),
                (agent.get("recently_sold") or {}).get("count"),
                (agent.get("for_sale_price") or {}).get("count"),
            ]):
                continue

            # Safe address access (some are null)
            addr = agent.get("address") or {}
            city = smart_title_loose(addr.get("city", ""))
            postal = addr.get("postal_code", "")

            # --------- KEEP last-activity logic EXACTLY as you had ----------
            first_year = agent.get("first_year")
            last_sale_date_raw = (agent.get("recently_sold") or {}).get("last_sold_date") \
                                 or (agent.get("for_sale_price") or {}).get("last_listing_date")
            last_sale_date = format_date(last_sale_date_raw)
            days_since_last = calculate_days_since(last_sale_date_raw) if last_sale_date_raw else None
            # ----------------------------------------------------------------

            # Validate first active year/month and compute experience
            y_valid, m_valid = validate_first_active(agent.get("first_year"), agent.get("first_month"))
            exp_years = compute_experience_years(y_valid, m_valid)

            # Name (title-case from first/last if available)
            first = agent.get("first_name") or ""
            last  = agent.get("last_name") or ""
            if first or last:
                full_name = smart_title(f"{first} {last}".strip())
            else:
                full_name = smart_title(agent.get("full_name", ""))

            # Phones (safe, normalized US)
            agent_phones = agent.get("phones") or []
            if agent_phones and isinstance(agent_phones[0], dict):
                phone_primary = normalize_us_phone(agent_phones[0].get("number"), agent_phones[0].get("ext"))
            else:
                phone_primary = ""

            office = agent.get("office") or {}
            office_phone = get_office_phone(office)

            # NEW: prettify office name and address
            office_name = prettify_office_name(office.get("name", ""))
            office_address = build_office_address(office)

            # Build row
            agent_row = {
                # SECTION 1: Scoring Features
                "state": usps,
                "agent_base_city": city,
                "agent_base_zipcode": postal,
                "agent_rating": agent.get("agent_rating"),
                "review_count": agent.get("review_count", 0),

                # SECTION 2: Activity Metrics (UNCHANGED date logic)
                "active_listings_count": (agent.get("for_sale_price") or {}).get("count", 0),
                "recently_sold_count": (agent.get("recently_sold") or {}).get("count", 0),
                "last_sale_date": last_sale_date,
                "days_since_last_sale": days_since_last,

                # SECTION 3: Price Data
                "active_listings_min_price": (agent.get("for_sale_price") or {}).get("min"),
                "active_listings_max_price": (agent.get("for_sale_price") or {}).get("max"),
                "recently_sold_min_price": (agent.get("recently_sold") or {}).get("min"),
                "recently_sold_max_price": (agent.get("recently_sold") or {}).get("max"),

                # SECTION 4: Geographic & Specialization
                "service_zipcodes": consolidate_list(agent.get("zips") or []),
                "service_areas": consolidate_list([ (a or {}).get("name") for a in (agent.get("served_areas") or []) ]),
                "marketing_area_cities": consolidate_list([ (c or {}).get("name") for c in (agent.get("marketing_area_cities") or []) ]),
                "specializations": consolidate_list([ (s or {}).get("name") for s in (agent.get("specializations") or []) ]),
                "agent_type": consolidate_list(agent.get("agent_type") or []),

                # SECTION 5: Experience (VALIDATED)
                "first_year_active": y_valid,
                "first_month_active": m_valid,
                "experience_years": exp_years,

                # SECTION 6: Geo/Compliance
                "is_realtor": bool(agent.get("is_realtor", False)),
                "designations": consolidate_designations(agent),
                "languages": consolidate_list(agent.get("languages") or ["English"]),

                # SECTION 7: Identity
                "advertiser_id": agent.get("advertiser_id"),
                "full_name": full_name,

                # SECTION 8: Display Fields (office prettified)
                "agent_title": agent.get("title", ""),
                "office_name": office_name,
                "office_address": office_address,
                "phone_primary": phone_primary,
                "office_phone": office_phone,
                "agent_website": normalize_url(agent.get("href", "")),
                "has_photo": bool(agent.get("has_photo", False)),
                "agent_photo_url": (agent.get("photo") or {}).get("href", ""),
                "agent_bio": (agent.get("description", "")[:500] if agent.get("description") else ""),
            }

            agents_list.append(agent_row)

    print(f"\nStates found: {', '.join(sorted(states_found))}")
    if not agents_list:
        print("✗ ERROR: No valid agents found in JSON file")
        return None

    df = pd.DataFrame(agents_list)
    print("\nValidating and cleaning data...")
    df = validate_and_clean(df)

    # Column order (35 columns)
    column_order = [
        # Primary identity & core address (top)
        "advertiser_id","full_name","state","agent_base_city","agent_base_zipcode",
        # Review/Rating
        "review_count","agent_rating",
        # Activity metrics
        "active_listings_count","recently_sold_count","last_sale_date","days_since_last_sale",
        # Price data
        "active_listings_min_price","active_listings_max_price",
        "recently_sold_min_price","recently_sold_max_price",
        # Geographic coverage & expertise
        "service_zipcodes","service_areas","marketing_area_cities",
        # Experience
        "first_year_active","first_month_active","experience_years",
        # Attributes / compliance
        "designations","languages","specializations",
        # Display fields (last)
        "agent_type","agent_title","is_realtor","office_name","office_address",
        "phone_primary","office_phone","agent_website","has_photo","agent_photo_url","agent_bio",
    ]
    df = df[column_order]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("\n" + "="*70)
    print("✓ CONVERSION SUCCESSFUL")
    print("="*70)
    print(f"Output file: {output_path}")
    print(f"Total agents: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("="*70)
    return df

if __name__ == "__main__":
    json_to_csv_agents(INPUT_JSON_PATH, OUTPUT_CSV_PATH)

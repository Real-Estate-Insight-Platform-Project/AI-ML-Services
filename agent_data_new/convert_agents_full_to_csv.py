# convert_agents_full_to_csv.py
import json
import pandas as pd

INPUT_FILE = "agents_by_state.json"
OUTPUT_FILE = "agents_full.csv"

# Exclude any path that starts with these prefixes (dotted path)
EXCLUDE_PREFIXES = (
    "lang.srp_content",   # drop huge UI copy block
    "social_media",       # skip this
)

def should_exclude(path: str) -> bool:
    """Return True if this dotted key path should be excluded."""
    return any(path.startswith(prefix) for prefix in EXCLUDE_PREFIXES)

def flatten(value, parent_key=""):
    """
    Recursively flatten dicts/lists into a flat dict with dotted keys.
    Lists get numeric indexes: phones.0.number, specializations.1.name, zips.0, etc.
    Scalars stay as-is.
    """
    items = {}

    if isinstance(value, dict):
        for k, v in value.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if should_exclude(new_key):
                continue
            items.update(flatten(v, new_key))
    elif isinstance(value, list):
        # If list of scalars -> join as comma string for compactness
        if all(not isinstance(el, (dict, list)) for el in value):
            items[parent_key] = ", ".join(map(str, value))
        else:
            # list of dicts/mixed -> index each element
            for i, el in enumerate(value):
                new_key = f"{parent_key}.{i}" if parent_key else str(i)
                items.update(flatten(el, new_key))
    else:
        items[parent_key] = value
    return items

def extract_all_agents(input_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for entry in data:
        state = entry.get("state")
        usps  = entry.get("usps")
        payload = entry.get("payload") or {}

        # Navigate safely to the agents array
        agents = (
            payload.get("data", {})
                   .get("pageData", {})
                   .get("agents", [])
        )

        for agent in agents:
            flat = flatten(agent)
            # add state/usps/context columns up front
            flat["state"] = state
            flat["usps"] = usps
            rows.append(flat)

    return rows

def main():
    rows = extract_all_agents(INPUT_FILE)
    if not rows:
        print("No agents found in input.")
        return

    # Build a DataFrame with all discovered columns
    df = pd.DataFrame(rows)

    # Move state/usps to the front if present
    front = [c for c in ["state", "usps"] if c in df.columns]
    other = [c for c in df.columns if c not in front]
    df = df[front + other]

    # Write CSV
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"✅ Wrote {len(df)} rows, {len(df.columns)} columns → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

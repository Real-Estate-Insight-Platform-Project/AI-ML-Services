import pandas as pd
import json
import os

def load_zip_series(csv_path: str, zip_col: str) -> pd.Series:
    df = pd.read_csv(csv_path, dtype={zip_col: str})
    z = (df[zip_col]
         .astype(str)
         .str.replace(r"\D", "", regex=True)
         .str.zfill(5))
    return z.dropna().drop_duplicates()

def write_rows_csv(rows: list[dict], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not rows:
        return
    df = pd.DataFrame(rows)
    # lists → JSON strings (CSV-safe)
    for c in ["reviews", "active_listing_urls", "phones"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: json.dumps(x, ensure_ascii=False))
    if not os.path.exists(out_csv):
        df.to_csv(out_csv, index=False)
    else:
        old = pd.read_csv(out_csv)
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["profile_url"], keep="last")
        merged.to_csv(out_csv, index=False)

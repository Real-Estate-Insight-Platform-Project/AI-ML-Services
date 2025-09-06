from supabase import create_client
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(url, key)

batch_size = 1000
offset = 0
all_data = []

while True:
    response = (
        supabase.table("State_Market")
        .select("*")
        .range(offset, offset + batch_size - 1)
        .execute()
    )
    data = response.data
    if not data:  # no more rows
        break
    all_data.extend(data)
    offset += batch_size

df = pd.DataFrame(all_data)
df.to_csv("State_Market.csv", index=False)
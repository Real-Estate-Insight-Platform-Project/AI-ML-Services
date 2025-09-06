from supabase import create_client
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(url, key)

# Replace "your_table" with the actual table name
response = supabase.table("State_Market").select("*").execute()

# Convert to DataFrame
data = response.data
df = pd.DataFrame(data)

df.to_csv("State_Market.csv", index=False)
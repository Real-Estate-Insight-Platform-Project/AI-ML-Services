import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

# Supabase connection settings
db_user = "postgres"   # default username in Supabase
db_pass = os.getenv("SUPABASE_DB_PASSWORD")  # find in Supabase dashboard
db_host = os.getenv("POSTGRES_HOST")      # e.g. db.abcd.supabase.co
db_port = "5432"
db_name = "postgres"   # default DB name

# Create connection string
connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

# Connect with SQLAlchemy
engine = create_engine(connection_string)

# Load full table into Pandas
df = pd.read_sql("SELECT * FROM State_Market", engine)

print(df.shape)
print(df.head())

# # Optionally, save to CSV
# df.to_csv("State_Market.csv", index=False)
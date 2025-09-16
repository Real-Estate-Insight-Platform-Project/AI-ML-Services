import requests
import os
import json
from supabase import create_client, Client
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv(), override=True)
# --- Fetch data from API ---
url = "https://us-realtor.p.rapidapi.com/api/v1/property/list"

querystring = {"location": "New Hampshire", "limit": "10",
               "state_code": "NH", "area_type": "state", "sort_by": "photo_count"}

headers = {
    "x-rapidapi-key": "52e25f44e9mshb7283b0e0f19e38p131f53jsnfed76e63d4fe",
    "x-rapidapi-host": "us-realtor.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)
data = response.json()

# print("API Response:")
# print(json.dumps(data, indent=2))
# --- Supabase setup ---
SUPABASE_URL = "https://kjyiuoqbkzmbdkrpwqcr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtqeWl1b3Fia3ptYmRrcnB3cWNyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzYxNDc5OCwiZXhwIjoyMDczMTkwNzk4fQ.8t60wee7Op3EU19phk2OoxQGD-vZ1Lzxau6c50dxDj0"  # load from env
# SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # load from env
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# if "data" not in data or "home_search" not in data["data"]:
#     print("Error: Unexpected API response format")
#     exit(1)
results = data["data"]["home_search"]["results"]

# --- Map function: JSON -> DB row ---


def map_property(item):
    address = item.get("location", {}).get("address", {})
    description = item.get("description", {})

    # bathrooms = full + half * 0.5
    bathrooms = None
    if description.get("baths_full") or description.get("baths_half"):
        bathrooms = (description.get("baths_full", 0) or 0) + \
            0.5 * (description.get("baths_half", 0) or 0)

    return {
        "title": f"{description.get('type', 'property').replace('_', ' ').title()} in {address.get('city', '')}",
        "description": f"{description.get('beds', '')} beds, {bathrooms or ''} baths",
        "price": item.get("list_price"),
        "address": address.get("line"),
        "city": address.get("city"),
        "state": address.get("state"),
        "zip_code": address.get("postal_code"),
        "property_type": "house" if description.get("type") == "single_family" else "apartment",
        "bedrooms": description.get("beds"),
        "bathrooms": bathrooms,
        "square_feet": description.get("sqft"),
        "lot_size": description.get("lot_sqft"),
        "year_built": None,
        "property_image": item.get("primary_photo", {}).get("href"),
        "property_hyperlink": item.get("href"),
        "longitude_coordinates": address.get("coordinate", {}).get("lon"),
        "latitude_coordinates": address.get("coordinate", {}).get("lat"),
    }


# --- Insert into Supabase ---
for item in results:
    row = map_property(item)
    response = supabase.table("properties").insert(row).execute()
    print("Inserted:", response.data)
    if response.error:
        print("Error:", response.error)

import os
import time
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict, Any, Optional

# --- Setup and Configuration ---

# Load environment variables from a .env file for security
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# RapidAPI configuration (using an environment variable is recommended)
RAPIDAPI_KEY = os.getenv(
    "RAPIDAPI_KEY", "52e25f44e9mshb7283b0e0f19e38p131f53jsnfed76e63d4fe")
RAPIDAPI_HOST = "us-realtor.p.rapidapi.com"

# --- Initialize Supabase client with error handling ---
supabase: Client = None  # Initialize as None
try:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print(" Error: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")
        sys.exit(1)  # Exit the script

    print("Attempting to connect to Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(" Successfully connected to Supabase.")

except Exception as e:
    print(f" Critical Error: Could not connect to Supabase.")
    print(f"   Please check your SUPABASE_URL in the .env file and your network connection.")
    print(f"   Details: {e}")
    sys.exit(1)  # Exit the script if connection fails


def fetch_properties_to_update(limit: int = 25) -> List[Dict[str, Any]]:
    """
    Fetches properties from Supabase that do not have a noise_score yet.
    We select the primary key 'id' for updating and the 'property_id' for the API call.
    """
    print(
        f"Fetching up to {limit} properties from the database that need surroundings data...")
    try:
        # We query for rows where 'noise_score' is NULL to find unprocessed properties.
        response = supabase.table('properties').select('id, property_id').is_(
            'noise_score', "NULL").limit(limit).execute()

        if response.data:
            print(f"Found {len(response.data)} properties to process.")
            return response.data
        else:
            print(" No properties found that need updating.")
            return []
    except Exception as e:
        print(f" Error fetching properties from Supabase: {e}")
        return []


def fetch_surrounding_data(property_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches noise, flood, and FEMA data for a given property_id from the RapidAPI.
    """
    url = f"https://{RAPIDAPI_HOST}/api/v1/property/get-surroundings"
    querystring = {"property_id": property_id}
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)

        api_response_data = response.json().get('data', {})

        if not api_response_data:
            print(
                f" No 'data' in API response for property_id {property_id}")
            return None

        # Safely extract the required information using .get() to avoid errors
        noise_score = api_response_data.get('noise', {}).get('score')
        flood_factor = api_response_data.get(
            'flood', {}).get('flood_factor_score')

        # FEMA zone is a list, so we safely get the first item
        fema_zone_list = api_response_data.get(
            'flood', {}).get('fema_zone', [])
        fema_zone = fema_zone_list[0] if fema_zone_list else None

        return {
            'noise_score': noise_score,
            'flood_factor': flood_factor,
            'fema_zone': fema_zone
        }

    except requests.exceptions.RequestException as e:
        print(f" API request failed for property_id {property_id}: {e}")
        return None
    except Exception as e:
        print(
            f" An unexpected error occurred while fetching API data for {property_id}: {e}")
        return None


def update_property_in_db(row_id: int, data_to_update: Dict[str, Any]) -> bool:
    """
    Updates a specific property row in Supabase using its primary key (id).
    """
    try:
        # Use the primary key 'id' to ensure we only update the correct row
        response = supabase.table('properties').update(
            data_to_update).eq('id', row_id).execute()

        if response.data:
            print(f" Successfully updated row ID {row_id} in the database.")
            return True
        else:
            error_message = response.get(
                "error", "Unknown error from Supabase")
            print(
                f" Failed to update row ID {row_id}. Reason: {error_message}")
            return False
    except Exception as e:
        print(f" An exception occurred while updating row ID {row_id}: {e}")
        return False


def main():
    """
    Main function to orchestrate fetching properties, getting surroundings data, and updating the database.
    """
    print("\n---  Starting Property Surroundings Update Script ---")

    # 1. Get the list of properties to work on
    properties_to_process = fetch_properties_to_update()

    if not properties_to_process:
        print("--- Script finished. No properties to update. ---")
        return

    success_count = 0
    error_count = 0

    for prop in properties_to_process:
        row_id = prop['id']
        api_property_id = prop['property_id']

        print(
            f"\nProcessing property with ID: {api_property_id} (Database Row: {row_id})")

        # 2. Fetch data from the RapidAPI for the current property
        surroundings_data = fetch_surrounding_data(api_property_id)

        # 3. If API call was successful, update the database
        if surroundings_data:
            if update_property_in_db(row_id, surroundings_data):
                success_count += 1
            else:
                error_count += 1
        else:
            print(
                f"Skipping database update for property {api_property_id} due to API fetch failure.")
            error_count += 1

        # Be a good API citizen: wait a moment before the next request to avoid rate limits
        time.sleep(1.5)

    print("\n---  Script Finished ---")
    print(f" Summary: {success_count} properties updated successfully.")
    print(f" Summary: {error_count} properties failed to update.")


if __name__ == "__main__":
    main()

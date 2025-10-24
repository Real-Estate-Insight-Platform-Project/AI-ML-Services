import os
import requests
import supabase
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configuration
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
RAPIDAPI_HOST = os.getenv('RAPIDAPI_HOST')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Initialize Supabase client
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_property_details(property_id):
    """Fetch property details from RapidAPI"""
    url = "https://us-realtor.p.rapidapi.com/api/v1/property/details"
    querystring = {"property_id": property_id}
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching property {property_id}: {e}")
        return None


def update_property_in_database(property_id, status, year_built):
    """Update property status and year_built in Supabase"""
    try:
        update_data = {
            "listing_status": status,
            "year_built": year_built
        }

        response = supabase_client.table("properties")\
            .update(update_data)\
            .eq("property_id", property_id)\
            .execute()

        if response.data:
            print(
                f"✓ Updated property {property_id}: status={status}, year_built={year_built}")
            return True
        else:
            print(f"✗ Failed to update property {property_id}")
            return False

    except Exception as e:
        print(f"Error updating property {property_id} in database: {e}")
        return False


def insert_price_log(property_id, last_price_change_date, last_price_change_amount):
    """Insert price change record into property_price_log table"""
    try:
        log_data = {
            "property_id": property_id,
            "last_price_change_date": last_price_change_date,
            "last_price_change_amount": last_price_change_amount
        }

        response = supabase_client.table("property_price_log")\
            .insert(log_data)\
            .execute()

        if response.data:
            print(f"✓ Added price log for property {property_id}")
            return True
        else:
            print(f"✗ Failed to add price log for property {property_id}")
            return False

    except Exception as e:
        print(f"Error inserting price log for property {property_id}: {e}")
        return False


def get_all_properties():
    """Fetch all properties from the database"""
    try:
        response = supabase_client.table("properties")\
            .select("property_id")\
            .execute()
        return response.data
    except Exception as e:
        print(f"Error fetching properties from database: {e}")
        return []


def process_properties():
    """Main function to process all properties"""
    print("Starting property update process...")

    # Get all properties from database
    properties = get_all_properties()

    if not properties:
        print("No properties found in the database")
        return

    print(f"Found {len(properties)} properties to process")

    success_count = 0
    error_count = 0

    for property_data in properties:
        property_id = property_data['property_id']
        print(f"\nProcessing property: {property_id}")

        # Fetch property details from RapidAPI
        api_response = get_property_details(property_id)

        if not api_response or 'data' not in api_response:
            print(f"✗ No data received for property {property_id}")
            error_count += 1
            continue

        home_data = api_response['data']['home']

        # Extract required fields
        status = home_data.get('status')
        year_built = home_data.get('description', {}).get('year_built')
        last_price_change_date = home_data.get('last_price_change_date')
        last_price_change_amount = home_data.get('last_price_change_amount')

        # Update property in database
        if status and year_built:
            update_success = update_property_in_database(
                property_id, status, year_built)

            if update_success:
                success_count += 1

                # If status is 'for_sale', add to price log
                if status == 'for_sale' and (last_price_change_date or last_price_change_amount):
                    insert_price_log(
                        property_id, last_price_change_date, last_price_change_amount)
            else:
                error_count += 1
        else:
            print(f"✗ Missing required data for property {property_id}")
            error_count += 1

    print(f"\n=== Process Complete ===")
    print(f"Successfully processed: {success_count} properties")
    print(f"Errors: {error_count} properties")
    print(f"Total: {len(properties)} properties")


if __name__ == "__main__":
    # Validate environment variables
    required_vars = ['RAPIDAPI_KEY', 'RAPIDAPI_HOST',
                     'SUPABASE_URL', 'SUPABASE_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
    else:
        process_properties()

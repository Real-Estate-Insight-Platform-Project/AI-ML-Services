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
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

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


def map_api_status_to_database_status(api_status):
    """Map API status values to database allowed values"""
    status_mapping = {
        'for_sale': 'active',
        'sold': 'sold',
        'pending': 'pending',
        'off_market': 'off_market',
        'for_rent': 'off_market',
        'coming_soon': 'active',
    }
    return status_mapping.get(api_status, 'off_market')


def update_property_in_database_fixed(property_id, status, year_built, price):
    """Fixed update function that handles RLS, empty responses, and price"""
    try:
        # Convert property_id to integer to match database type
        property_id_int = int(property_id)

        # Map API status to database status
        mapped_status = map_api_status_to_database_status(status)

        update_data = {
            "listing_status": mapped_status
        }

        # Only add non-None values to the update payload
        if year_built is not None:
            update_data["year_built"] = year_built
        if price is not None:
            # Assuming price in DB is numeric
            update_data["price"] = float(price)

        print(f"Updating property {property_id_int} with: {update_data}")

        # Perform the update
        response = supabase_client.table("properties")\
            .update(update_data)\
            .eq("property_id", property_id_int)\
            .execute()

        # In Supabase, successful updates often return empty array []
        # We need to verify the update worked by checking the row afterwards

        # Check if the update actually worked by reading the row back
        verify_response = supabase_client.table("properties")\
            .select("property_id, listing_status, year_built, price")\
            .eq("property_id", property_id_int)\
            .execute()

        if verify_response.data:
            current_data = verify_response.data[0]

            # Check if our updates are reflected
            update_successful = True

            if current_data.get('listing_status') != mapped_status:
                update_successful = False
                print(
                    f"  FAILED: listing_status mismatch. DB: {current_data.get('listing_status')}, Expected: {mapped_status}")

            if "year_built" in update_data and current_data.get('year_built') != update_data["year_built"]:
                update_successful = False
                print(
                    f"  FAILED: year_built mismatch. DB: {current_data.get('year_built')}, Expected: {update_data['year_built']}")

            # Compare price as float
            if "price" in update_data:
                db_price = current_data.get('price')
                # Handle potential type difference (e.g., DB returns '123.00')
                if db_price is None or float(db_price) != update_data["price"]:
                    update_successful = False
                    print(
                        f"  FAILED: price mismatch. DB: {db_price}, Expected: {update_data['price']}")

            if update_successful:
                print(f"SUCCESS: Verified update for property {property_id}")
                return True
            else:
                print(f"PARTIAL/FAILED: Update did not fully apply. See failures above.")
                return False
        else:
            print(
                f"ERROR: Cannot verify update - property not found after update attempt")
            return False

    except Exception as e:
        print(f"Error updating property {property_id} in database: {e}")
        return False


def insert_price_log_fixed(property_id, last_price_change_date, last_price_change_amount):
    """Fixed price log insertion"""
    try:
        # Convert property_id to integer
        property_id_int = int(property_id)

        # Only insert if we have at least one of the values
        if last_price_change_date is None and last_price_change_amount is None:
            print(f"SKIPPED: No price change data for property {property_id}")
            return True

        log_data = {
            "property_id": property_id_int,
            # Supabase client handles None values correctly
            "last_price_change_date": last_price_change_date,
            "last_price_change_amount": last_price_change_amount
        }

        # Handle cases where only one value is present
        if last_price_change_date is None:
            # Use a default or handle as needed; here we just pass None
            pass

        # If date is missing but amount is present, you might want to default the date
        # For this example, we assume the DB schema allows null dates or has a default

        # **** IMPORTANT: This insert will FAIL until you fix your
        # **** property_price_log table schema. See explanation below.

        response = supabase_client.table("property_price_log")\
            .insert(log_data)\
            .execute()

        # For inserts, we should get data back if successful
        if response.data:
            print(f"SUCCESS: Added price log for property {property_id}")
            return True
        else:
            print(
                f"FAILED: No data returned from price log insert for property {property_id}")
            if hasattr(response, 'error') and response.error:
                print(f"ERROR: {response.error}")
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


def process_properties_fixed():
    """Main function to process all properties with fixed update logic"""
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
            print(f"ERROR: No data received for property {property_id}")
            error_count += 1
            continue

        home_data = api_response['data'].get('home')

        if not home_data:
            print(f"ERROR: No home data for property {property_id}")
            error_count += 1
            continue

        # Extract required fields
        status = home_data.get('status')
        year_built = home_data.get('description', {}).get('year_built')
        price = home_data.get('list_price')  # <-- ADDED THIS
        last_price_change_date = home_data.get('last_price_change_date')
        last_price_change_amount = home_data.get('last_price_change_amount')

        # Validate required fields
        if not status:
            print(f"ERROR: Missing status for property {property_id}")
            error_count += 1
            continue

        # Update property in database
        update_success = update_property_in_database_fixed(
            property_id, status, year_built, price)  # <-- ADDED price

        if update_success:
            success_count += 1

            # If status is 'for_sale' (mapped to 'active'), add to price log
            mapped_status = map_api_status_to_database_status(status)
            if mapped_status == 'active':
                price_log_success = insert_price_log_fixed(
                    property_id, last_price_change_date, last_price_change_amount
                )
                if not price_log_success:
                    print(
                        f"WARNING: Price log failed for property {property_id}")
        else:
            error_count += 1

    print(f"\n=== Process Complete ===")
    print(f"Successfully processed: {success_count} properties")
    print(f"Errors: {error_count} properties")
    print(f"Total: {len(properties)} properties")


def check_rls_status():
    """Check if RLS is enabled on the tables"""
    print("=== CHECKING RLS STATUS ===")
    try:
        # Try to check if we can update without RLS issues
        test_property_id = 1000140996  # Using one from your log
        test_year = 9999

        # Try a simple update
        test_update = supabase_client.table("properties")\
            .update({"year_built": test_year})\
            .eq("property_id", test_property_id)\
            .execute()

        print(f"Update response: {test_update}")

        # Verify the update
        updated_data = supabase_client.table("properties")\
            .select("property_id, year_built")\
            .eq("property_id", test_property_id)\
            .execute()

        print(f"After update attempt: {updated_data.data}")

        if updated_data.data and updated_data.data[0]['year_built'] == test_year:
            print("RLS check PASSED: Update was successful.")
            # Revert the change
            supabase_client.table("properties").update({"year_built": None}).eq(
                "property_id", test_property_id).execute()
            print("Reverted test data.")
            return True
        else:
            print("RLS check FAILED: Data was not updated.")
            return False

    except Exception as e:
        print(f"RLS check error: {e}")
        return False


if __name__ == "__main__":
    # Validate environment variables
    required_vars = ['RAPIDAPI_KEY', 'RAPIDAPI_HOST',
                     'SUPABASE_URL', 'SUPABASE_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
    else:
        print("Checking RLS status first...")
        if check_rls_status():
            print("\nStarting main process...")
            process_properties_fixed()
        else:
            print("\nRLS check failed. Your updates will not work.")
            print("You must either:")
            print(
                "  1. Ensure SUPABASE_KEY is your 'service_role' key (from Project Settings > API).")
            print(
                "  2. Or, disable RLS on the 'properties' table in Supabase (Authentication > Policies).")
            print(
                "  3. Or, create an RLS policy that allows this script to perform updates.")

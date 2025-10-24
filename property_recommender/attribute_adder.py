import requests
import json
from supabase import create_client, Client
from datetime import datetime
import os
from typing import Dict, Any, Optional, List
import time
from dotenv import load_dotenv

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
RAPIDAPI_HOST = os.getenv('RAPIDAPI_HOST')

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_property_ids_from_database(batch_size: int = 100, offset: int = 0):
    """Fetch property IDs from the database that have valid property_id and need surroundings data"""
    try:
        # First approach: Try to filter by valid property_id using different methods
        # Since property_id is bigint, we only need to check for NOT NULL
        response = supabase.table('properties') \
            .select('id, property_id, noise_score, flood_factor, fema_zone') \
            .not_.is_('property_id', 'null') \
            .limit(batch_size) \
            .offset(offset) \
            .execute()

        print(
            f"Found {len(response.data)} properties with non-null property_id")

        # Filter out any properties that might have invalid property_id values
        valid_properties = []
        for prop in response.data:
            prop_id = prop.get('property_id')
            # Ensure property_id is a valid integer/string that can be used in API calls
            # Check if it's not None and not empty string
            if prop_id and str(prop_id).strip():
                valid_properties.append(prop)
            else:
                print(f"Skipping invalid property_id: {prop_id}")

        print(
            f"After filtering, {len(valid_properties)} valid properties remain")
        return valid_properties

    except Exception as e:
        print(f"Error fetching property IDs from database: {e}")
        # Fallback: try basic query without any filters
        try:
            response = supabase.table('properties') \
                .select('id, property_id') \
                .limit(batch_size) \
                .offset(offset) \
                .execute()

            # Manually filter for valid property_ids
            valid_properties = []
            for prop in response.data:
                prop_id = prop.get('property_id')
                if prop_id and str(prop_id).strip():
                    valid_properties.append(prop)

            print(
                f"Fetched {len(valid_properties)} valid properties (fallback method)")
            return valid_properties
        except Exception as e2:
            print(f"Error with fallback query: {e2}")
            return []


def check_columns_exist():
    """Check if the required columns exist in the table"""
    try:
        # Try to select one record with the new columns
        test = supabase.table('properties') \
            .select('noise_score, flood_factor, fema_zone') \
            .limit(1) \
            .execute()
        print("Surroundings columns exist in the table")
        return True
    except Exception as e:
        print(f"Surroundings columns might not exist: {e}")
        return False


def get_surroundings_data(property_id: str):
    """Fetch surroundings data from the API for a specific property"""
    url = "https://us-realtor.p.rapidapi.com/api/v1/property/get-surroundings"

    # Ensure property_id is string for the API call
    querystring = {"property_id": str(property_id)}

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }

    try:
        print(f"Fetching surroundings data for property: {property_id}")
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()
        print(f"API response received for {property_id}")
        return data
    except requests.exceptions.RequestException as e:
        print(
            f"Error fetching surroundings data for property {property_id}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for property {property_id}: {e}")
        return None


def parse_surroundings_data(api_data: Dict[str, Any]):
    """Parse the API response and extract noise score, flood factor, and FEMA zone"""
    if not api_data:
        print("No API data provided")
        return None, None, None

    if 'data' not in api_data:
        print("No 'data' key in API response")
        return None, None, None

    data = api_data['data']

    # Extract noise score
    noise_score = None
    if 'noise' in data and 'score' in data['noise']:
        noise_score = data['noise']['score']
        print(f"Extracted noise_score: {noise_score}")
    else:
        print("No noise data found in response")

    # Extract flood factor score
    flood_factor = None
    if 'flood' in data and 'flood_factor_score' in data['flood']:
        flood_factor = data['flood']['flood_factor_score']
        print(f"Extracted flood_factor: {flood_factor}")
    else:
        print("No flood factor data found in response")

    # Extract FEMA zone
    fema_zone = None
    if 'flood' in data and 'fema_zone' in data['flood'] and data['flood']['fema_zone']:
        # Convert list to string if there are multiple zones
        fema_zone = ', '.join(data['flood']['fema_zone'])
        print(f"Extracted fema_zone: {fema_zone}")
    else:
        print("No FEMA zone data found in response")

    return noise_score, flood_factor, fema_zone


def update_property_in_database(property_id: str, noise_score: int, flood_factor: int, fema_zone: str):
    """Update the property record in the database with surroundings data"""
    try:
        update_data = {
            'updated_at': datetime.now().isoformat()
        }

        # Add the surroundings data if they exist
        if noise_score is not None:
            update_data['noise_score'] = noise_score
        if flood_factor is not None:
            update_data['flood_factor'] = flood_factor
        if fema_zone is not None:
            update_data['fema_zone'] = fema_zone

        print(f"Updating property {property_id} with data: {update_data}")

        response = supabase.table('properties') \
            .update(update_data) \
            .eq('property_id', property_id) \
            .execute()

        if response.data:
            print(f"Successfully updated property {property_id}")
            return True
        else:
            print(
                f"Failed to update property {property_id} - no data returned")
            return False

    except Exception as e:
        print(f"Error updating property {property_id} in database: {e}")
        return False


def process_single_property(property_record: Dict[str, Any]):
    """Process a single property: fetch surroundings data and update database"""
    property_id = property_record['property_id']
    db_id = property_record['id']

    print(
        f"\n=== Processing property ID: {property_id} (Database ID: {db_id}) ===")

    # Ensure property_id is valid
    if not property_id or str(property_id).strip() == '':
        print(f"Invalid property_id: {property_id}")
        return False

    # Fetch surroundings data from API
    surroundings_data = get_surroundings_data(property_id)

    if not surroundings_data:
        print(f"No surroundings data received for property {property_id}")
        return False

    # Parse the data
    noise_score, flood_factor, fema_zone = parse_surroundings_data(
        surroundings_data)

    print(
        f"Summary for {property_id}: Noise: {noise_score}, Flood: {flood_factor}, FEMA: {fema_zone}")

    # Update the database
    success = update_property_in_database(
        property_id, noise_score, flood_factor, fema_zone)

    return success


def process_all_properties(batch_size: int = 50, delay_between_requests: float = 1.0):
    """Process all properties in the database that have valid property_id and need surroundings data"""
    offset = 0
    total_processed = 0
    total_success = 0
    total_failed = 0

    print("Starting to process properties for surroundings data...")

    # First, check if columns exist
    columns_exist = check_columns_exist()

    while True:
        # Fetch a batch of properties
        properties = fetch_property_ids_from_database(batch_size, offset)

        if not properties:
            print("No more properties to process.")
            break

        print(
            f"\n*** Processing batch of {len(properties)} properties (offset: {offset}) ***")

        for i, property_record in enumerate(properties, 1):
            print(f"\n--- Property {i}/{len(properties)} in current batch ---")

            # Double-check that property_id is valid
            if not property_record.get('property_id') or str(property_record.get('property_id')).strip() == '':
                print(
                    f"Skipping property with invalid property_id: {property_record}")
                total_failed += 1
                continue

            success = process_single_property(property_record)

            if success:
                total_success += 1
            else:
                total_failed += 1

            total_processed += 1

            # Add delay between API requests to avoid rate limiting
            if i < len(properties):  # Don't delay after the last property
                print(
                    f"Waiting {delay_between_requests} seconds before next request...")
                time.sleep(delay_between_requests)

        # Move to next batch
        offset += batch_size

        # Brief pause between batches
        print("Batch completed. Taking a short break...")
        time.sleep(0.5)

    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total processed: {total_processed}")
    print(f"Successfully updated: {total_success}")
    print(f"Failed: {total_failed}")

    return total_processed, total_success, total_failed


def test_with_single_property():
    """Test with a single known property ID"""
    test_property_id = "3034292651"  # From your example

    print(f"Testing with single property: {test_property_id}")

    # First, check if this property exists in our database
    try:
        response = supabase.table('properties') \
            .select('id, property_id') \
            .eq('property_id', test_property_id) \
            .execute()

        if response.data:
            property_record = response.data[0]
            success = process_single_property(property_record)
            print(f"Test result: {'SUCCESS' if success else 'FAILED'}")
        else:
            print(f"Test property {test_property_id} not found in database")

    except Exception as e:
        print(f"Error during test: {e}")


def get_property_stats():
    """Get statistics about properties in the database"""
    try:
        # Total properties
        total_response = supabase.table('properties') \
            .select('id', count='exact') \
            .execute()

        print(f"\n=== DATABASE STATISTICS ===")
        print(f"Total properties: {total_response.count}")

        # Try to get count of properties with surroundings data (might fail if columns don't exist)
        try:
            with_noise_data = supabase.table('properties') \
                .select('id', count='exact') \
                .not_.is_('noise_score', 'null') \
                .execute()
            print(f"Properties with noise data: {with_noise_data.count}")
        except:
            print(
                "Properties with noise data: Could not determine (column might not exist)")

        try:
            with_flood_data = supabase.table('properties') \
                .select('id', count='exact') \
                .not_.is_('flood_factor', 'null') \
                .execute()
            print(f"Properties with flood data: {with_flood_data.count}")
        except:
            print(
                "Properties with flood data: Could not determine (column might not exist)")

    except Exception as e:
        print(f"Error getting basic statistics: {e}")


def main():
    """Main function to orchestrate the surroundings data processing"""

    print("=== PROPERTY SURROUNDINGS DATA PROCESSOR ===")

    # Show basic statistics first
    get_property_stats()

    # Option 0: Test with a single property first
    print("\n1. Testing with single property...")
    test_with_single_property()

    # Option 1: Process all properties with valid property_id
    print("\n2. Processing all properties with valid property_id...")
    process_all_properties(batch_size=20, delay_between_requests=1.0)

    # Show final statistics
    print("\n3. Final statistics:")
    get_property_stats()


if __name__ == "__main__":
    main()
# this is correct one

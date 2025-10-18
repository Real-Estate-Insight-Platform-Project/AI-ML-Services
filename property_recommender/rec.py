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

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# List of all US states and territories with their abbreviations
US_STATES = [
    {"name": "Alabama", "abbr": "AL"},
    {"name": "Alaska", "abbr": "AK"},
    {"name": "Arizona", "abbr": "AZ"},
    {"name": "Arkansas", "abbr": "AR"},
    {"name": "California", "abbr": "CA"},
    {"name": "Colorado", "abbr": "CO"},
    {"name": "Connecticut", "abbr": "CT"},
    {"name": "Delaware", "abbr": "DE"},
    {"name": "Florida", "abbr": "FL"},
    {"name": "Georgia", "abbr": "GA"},
    {"name": "Hawaii", "abbr": "HI"},
    {"name": "Idaho", "abbr": "ID"},
    {"name": "Illinois", "abbr": "IL"},
    {"name": "Indiana", "abbr": "IN"},
    {"name": "Iowa", "abbr": "IA"},
    {"name": "Kansas", "abbr": "KS"},
    {"name": "Kentucky", "abbr": "KY"},
    {"name": "Louisiana", "abbr": "LA"},
    {"name": "Maine", "abbr": "ME"},
    {"name": "Maryland", "abbr": "MD"},
    {"name": "Massachusetts", "abbr": "MA"},
    {"name": "Michigan", "abbr": "MI"},
    {"name": "Minnesota", "abbr": "MN"},
    {"name": "Mississippi", "abbr": "MS"},
    {"name": "Missouri", "abbr": "MO"},
    {"name": "Montana", "abbr": "MT"},
    {"name": "Nebraska", "abbr": "NE"},
    {"name": "Nevada", "abbr": "NV"},
    {"name": "New Hampshire", "abbr": "NH"},
    {"name": "New Jersey", "abbr": "NJ"},
    {"name": "New Mexico", "abbr": "NM"},
    {"name": "New York", "abbr": "NY"},
    {"name": "North Carolina", "abbr": "NC"},
    {"name": "North Dakota", "abbr": "ND"},
    {"name": "Ohio", "abbr": "OH"},
    {"name": "Oklahoma", "abbr": "OK"},
    {"name": "Oregon", "abbr": "OR"},
    {"name": "Pennsylvania", "abbr": "PA"},
    {"name": "Rhode Island", "abbr": "RI"},
    {"name": "South Carolina", "abbr": "SC"},
    {"name": "South Dakota", "abbr": "SD"},
    {"name": "Tennessee", "abbr": "TN"},
    {"name": "Texas", "abbr": "TX"},
    {"name": "Utah", "abbr": "UT"},
    {"name": "Vermont", "abbr": "VT"},
    {"name": "Virginia", "abbr": "VA"},
    {"name": "Washington", "abbr": "WA"},
    {"name": "West Virginia", "abbr": "WV"},
    {"name": "Wisconsin", "abbr": "WI"},
    {"name": "Wyoming", "abbr": "WY"},
    {"name": "District of Columbia", "abbr": "DC"},
    {"name": "Puerto Rico", "abbr": "PR"},
    {"name": "Guam", "abbr": "GU"},
    {"name": "American Samoa", "abbr": "AS"},
    {"name": "U.S. Virgin Islands", "abbr": "VI"},
    {"name": "Northern Mariana Islands", "abbr": "MP"}
]

# List of sorting options to iterate through
SORT_OPTIONS = [
    "photo_count",
    "relevance",
    "newest",
    "lowest_price",
    "highest_price",
    "open_house_date",
    "price_reduced",
    "largest_sqft",
    "lot_size"
]


def fetch_property_data(state_abbr: str, state_name: str, sort_by: str, offset: int = 0):
    """Fetch property data from the Realtor API for a specific state"""
    url = "https://us-realtor.p.rapidapi.com/api/v1/property/list"
    querystring = {
        "location": state_name,
        "limit": "50",
        "offset": str(offset),
        "state_code": state_abbr,
        "area_type": "state",
        "sort_by": sort_by
    }
    headers = {
        "x-rapidapi-key": "52e25f44e9mshb7283b0e0f19e38p131f53jsnfed76e63d4fe",
        "x-rapidapi-host": "us-realtor.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {state_name} (Sort: {sort_by}): {e}")
        return None


def fetch_and_save_property_photos(property_db_id: int, property_api_id: str):
    """Fetch additional photos for a property and update the database record."""
    print(
        f"Fetching additional photos for Property API ID: {property_api_id}...")

    url = "https://realtor16.p.rapidapi.com/property/photos"
    querystring = {"property_id": property_api_id}
    headers = {
        "x-rapidapi-key": "52e25f44e9mshb7283b0e0f19e38p131f53jsnfed76e63d4fe",
        "x-rapidapi-host": "realtor16.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        photos_data = response.json()

        if not photos_data or 'data' not in photos_data or 'home_photos' not in photos_data['data']:
            print(
                f"No valid photo data structure returned for {property_api_id}")
            return

        # The actual image list is nested inside the response
        image_list = photos_data['data']['home_photos']

        if not isinstance(image_list, list):
            print(f"Photo data for {property_api_id} is not a list.")
            return

        # Extract all 'href' URLs from the list of photo dictionaries
        image_urls = [photo['href'] for photo in image_list if 'href' in photo]

        if not image_urls:
            print(f"No image URLs found in the response for {property_api_id}")
            return

        # Join the URLs into a single comma-separated string
        all_image_urls_str = ",".join(image_urls)

        # Update the property record in Supabase
        update_response = supabase.table('properties').update({
            'property_image': all_image_urls_str
        }).eq('id', property_db_id).execute()

        if update_response.data:
            print(
                f"Successfully saved {len(image_urls)} photo URLs for property DB ID {property_db_id}")
        else:
            print(
                f"Failed to update photo URLs for property DB ID {property_db_id}. Error: {update_response.error}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching photos for {property_api_id}: {e}")
    except (KeyError, TypeError) as e:
        print(f"Error parsing photo response for {property_api_id}: {e}")
    except Exception as e:
        print(
            f"An unexpected error occurred during photo fetching for {property_api_id}: {e}")


def transform_property_data(api_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Transform API data to match database schema"""
    if not api_data or 'data' not in api_data or 'home_search' not in api_data['data'] or 'results' not in api_data['data']['home_search']:
        return None

    results = api_data['data']['home_search']['results']
    transformed_properties = []

    for result in results:
        try:
            property_data = {
                'title': f"{result['description']['type'].replace('_', ' ').title()} {result['location']['address']['street_name']} {result['location']['address']['city']}",
                'description': f"{result['description']['beds']} bed, {result['description']['baths']} bath {result['description']['type'].replace('_', ' ').title()}",
                'price': float(result['list_price']),
                'address': result['location']['address']['line'],
                'city': result['location']['address']['city'],
                'state': result['location']['address']['state_code'],
                'zip_code': result['location']['address']['postal_code'],
                'property_type': map_property_type(result['description']['type']),
                'bedrooms': result['description']['beds'],
                'bathrooms': float(result['description']['baths']),
                'square_feet': result['description']['sqft'],
                'lot_size': float(result['description']['lot_sqft']) if result['description']['lot_sqft'] else None,
                'year_built': result['description'].get('year_built'),
                'listing_status': map_listing_status(result['status']),
                'property_image': result['primary_photo']['href'] if result.get('primary_photo') else None,
                'property_hyperlink': result['href'],
                'longitude_coordinates': float(result['location']['address']['coordinate']['lon']),
                'latitude_coordinates': float(result['location']['address']['coordinate']['lat']),
                'property_id': result['property_id'],
                'listed_date': result['list_date'],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            transformed_properties.append(property_data)
        except (KeyError, TypeError) as e:
            print(f"Missing key or wrong type in API response: {e}")
            continue
    return transformed_properties


def map_property_type(api_type: str) -> str:
    """Map API property type to database property type"""
    type_mapping = {
        'single_family': 'house',
        'apartment': 'apartment',
        'condo': 'condo',
        'townhouse': 'townhouse',
        'commercial': 'commercial'
    }
    return type_mapping.get(api_type, 'house')


def map_listing_status(api_status: str) -> str:
    """Map API listing status to database listing status"""
    status_mapping = {
        'for_sale': 'active',
        'sold': 'sold',
        'pending': 'pending',
        'off_market': 'off_market'
    }
    return status_mapping.get(api_status, 'active')


def save_to_database(properties: list):
    """Save properties to Supabase database and then fetch additional photos."""
    if not properties:
        print("No properties to save")
        return 0, 0

    success_count = 0
    error_count = 0

    for property_data in properties:
        try:
            #  Check if property already exists using the unique property_id
            existing_property = supabase.table('properties').select('id').eq(
                'property_id', property_data['property_id']).execute()

            if existing_property.data:
                print(
                    f"Property already exists: {property_data['title']} (Property ID: {property_data['property_id']})")
                continue

            # Insert into Supabase if it doesn't exist
            response = supabase.table('properties').insert(
                property_data).execute()

            if response.data:
                success_count += 1
                print(
                    f"Successfully saved property: {property_data['title']} (Property ID: {property_data['property_id']})")

                # After saving, fetch all photos
                new_property_record = response.data[0]
                property_db_id = new_property_record['id']
                property_api_id = new_property_record['property_id']

                # Add a small delay before fetching photos to be kind to the APIs
                time.sleep(0.5)

                fetch_and_save_property_photos(property_db_id, property_api_id)

            else:
                error_count += 1
                print(
                    f"Failed to save property: {property_data['title']} (Property ID: {property_data['property_id']}). Error: {response.error}")

        except Exception as e:
            error_count += 1
            print(
                f"Error saving property {property_data.get('title', 'N/A')} (Property ID: {property_data.get('property_id', 'N/A')}): {e}")

    print(
        f"Summary: {success_count} new properties saved, {error_count} failed")
    return success_count, error_count


def process_state(state: dict, sort_by: str, max_properties: int = 200):
    """Process properties for a single state with a specific sort order"""
    state_name = state["name"]
    state_abbr = state["abbr"]

    print(
        f"\nProcessing state: {state_name} ({state_abbr}) | Sort by: {sort_by}")

    total_saved = 0
    total_errors = 0
    offset = 0

    while total_saved < max_properties:
        print(f"Fetching properties with offset {offset}...")

        api_data = fetch_property_data(state_abbr, state_name, sort_by, offset)

        if not api_data:
            print(f"No data returned for {state_name}")
            break

        properties = transform_property_data(api_data)
        if not properties:
            print(f"No properties found for {state_name} at offset {offset}")
            break

        print(f"Found {len(properties)} properties for {state_name}")
        saved, errors = save_to_database(properties)
        total_saved += saved
        total_errors += errors

        if len(properties) < 50:
            break
        offset += 50
        time.sleep(1)

    print(
        f"Finished processing {state_name} for sort '{sort_by}'. Total new saved: {total_saved}, Total errors: {total_errors}")
    return total_saved, total_errors


def main():
    """Main function to orchestrate the process for all states and sort options"""
    total_properties_saved = 0
    total_errors = 0

    for state in US_STATES:
        for sort_option in SORT_OPTIONS:
            saved, errors = process_state(state, sort_option)
            total_properties_saved += saved
            total_errors += errors
            time.sleep(1)

        print(
            f"\n--- Finished all sorts for {state['name']}. Moving to next state. ---\n")
        time.sleep(2)

    print(f"\n======= FINAL SUMMARY =======")
    print(
        f"Total new properties saved across all states and sort options: {total_properties_saved}")
    print(f"Total errors encountered: {total_errors}")


if __name__ == "__main__":
    main()

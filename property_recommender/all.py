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

# Available sorting methods
SORTING_METHODS = [
    "relevance",
    "newest",
    "lowest_price",
    "highest_price",
    "open_house_date",
    "price_reduced",
    "largest_sqft",
    "lot_size",
    "photo_count"
]


def fetch_property_data(state_abbr: str, state_name: str, offset: int = 0, sort_by: str = "photo_count"):
    """Fetch property data from the Realtor API for a specific state with sorting"""
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
        print(
            f"Error fetching data for {state_name} with sort '{sort_by}': {e}")
        return None


def transform_property_data(api_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Transform API data to match database schema"""
    if not api_data or 'data' not in api_data:
        return None

    # Check if results exist in the response
    if 'home_search' not in api_data['data'] or 'results' not in api_data['data']['home_search']:
        return None

    results = api_data['data']['home_search']['results']
    transformed_properties = []

    for result in results:
        try:
            # Extract and transform data
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
                'year_built': None,  # Not available in API response
                'listing_status': map_listing_status(result['status']),
                'property_image': result['primary_photo']['href'] if result['primary_photo'] else None,
                'property_hyperlink': result['href'],
                'longitude_coordinates': float(result['location']['address']['coordinate']['lon']),
                'latitude_coordinates': float(result['location']['address']['coordinate']['lat']),
                'property_id': result['property_id'],
                'listed_date': result['list_date'],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'sort_method': sort_by  # Track which sorting method was used
            }

            transformed_properties.append(property_data)

        except KeyError as e:
            print(f"Missing key in API response: {e}")
            continue
        except Exception as e:
            print(f"Error transforming property data: {e}")
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
    return type_mapping.get(api_type, 'house')  # Default to 'house'


def map_listing_status(api_status: str) -> str:
    """Map API listing status to database listing status"""
    status_mapping = {
        'for_sale': 'active',
        'sold': 'sold',
        'pending': 'pending',
        'off_market': 'off_market'
    }
    return status_mapping.get(api_status, 'active')  # Default to 'active'


def save_to_database(properties: list):
    """Save properties to Supabase database"""
    if not properties:
        print("No properties to save")
        return 0, 0

    success_count = 0
    error_count = 0

    for property_data in properties:
        try:
            # Check if property already exists using the unique property_id
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
            else:
                error_count += 1
                print(
                    f"Failed to save property: {property_data['title']} (Property ID: {property_data['property_id']})")

        except Exception as e:
            error_count += 1
            print(
                f"Error saving property {property_data['title']} (Property ID: {property_data['property_id']}): {e}")

    print(
        f"Summary: {success_count} properties saved successfully, {error_count} failed")
    return success_count, error_count


def process_state_with_sorting(state: dict, sort_by: str = "photo_count", max_properties: int = 200):
    """Process properties for a single state with specific sorting method"""
    state_name = state["name"]
    state_abbr = state["abbr"]

    print(
        f"\nProcessing state: {state_name} ({state_abbr}) with sorting: '{sort_by}'")

    total_saved = 0
    total_errors = 0
    offset = 0

    while total_saved < max_properties:
        print(
            f"Fetching properties with offset {offset} and sort '{sort_by}'...")

        # Fetch data from API with specific sorting
        api_data = fetch_property_data(state_abbr, state_name, offset, sort_by)

        if not api_data:
            print(f"No data returned for {state_name} with sort '{sort_by}'")
            break

        # Transform data
        properties = transform_property_data(api_data)

        if not properties:
            print(
                f"No properties found for {state_name} at offset {offset} with sort '{sort_by}'")
            break

        print(
            f"Found {len(properties)} properties for {state_name} with sort '{sort_by}'")

        # Save to database
        saved, errors = save_to_database(properties)
        total_saved += saved
        total_errors += errors

        # Check if we've reached the end of results
        if len(properties) < 50:  # If we got fewer than the limit, we're at the end
            break

        # Increment offset for next batch
        offset += 50

        # Add a delay to avoid hitting API rate limits
        time.sleep(1)

    print(
        f"Finished processing {state_name} with sort '{sort_by}'. Total saved: {total_saved}, Total errors: {total_errors}")
    return total_saved, total_errors


def process_state_all_sorting_methods(state: dict, max_properties_per_sort: int = 100):
    """Process properties for a single state using all sorting methods"""
    total_saved = 0
    total_errors = 0

    for sort_method in SORTING_METHODS:
        saved, errors = process_state_with_sorting(
            state, sort_method, max_properties_per_sort)
        total_saved += saved
        total_errors += errors

        # Add delay between different sorting methods
        time.sleep(2)

    return total_saved, total_errors


def main_single_sorting(sort_by: str = "photo_count", max_properties_per_state: int = 100):
    """Main function to process all states with a single sorting method"""
    total_properties_saved = 0
    total_errors = 0

    print(f"Starting data collection with sorting method: '{sort_by}'")

    for state in US_STATES:
        saved, errors = process_state_with_sorting(
            state, sort_by, max_properties_per_state)
        total_properties_saved += saved
        total_errors += errors

        # Add a longer delay between states to avoid API rate limits
        time.sleep(2)

    print(f"\nFinal Summary for sorting '{sort_by}':")
    print(
        f"Total properties saved across all states: {total_properties_saved}")
    print(f"Total errors: {total_errors}")


def main_all_sorting_methods(max_properties_per_sort: int = 50):
    """Main function to process all states with all sorting methods"""
    total_properties_saved = 0
    total_errors = 0

    print("Starting data collection with ALL sorting methods")

    for state in US_STATES:
        saved, errors = process_state_all_sorting_methods(
            state, max_properties_per_sort)
        total_properties_saved += saved
        total_errors += errors

        # Add a longer delay between states to avoid API rate limits
        time.sleep(3)

    print(f"\nFinal Summary for ALL sorting methods:")
    print(
        f"Total properties saved across all states: {total_properties_saved}")
    print(f"Total errors: {total_errors}")


def main_custom_sorting(sorting_methods: List[str], max_properties_per_sort: int = 100):
    """Main function to process with custom sorting methods"""
    total_properties_saved = 0
    total_errors = 0

    print(
        f"Starting data collection with custom sorting methods: {sorting_methods}")

    for state in US_STATES:
        state_saved = 0
        state_errors = 0

        for sort_method in sorting_methods:
            saved, errors = process_state_with_sorting(
                state, sort_method, max_properties_per_sort)
            state_saved += saved
            state_errors += errors
            time.sleep(2)  # Delay between sorting methods

        total_properties_saved += state_saved
        total_errors += state_errors

        # Delay between states
        time.sleep(3)

    print(f"\nFinal Summary for custom sorting methods {sorting_methods}:")
    print(
        f"Total properties saved across all states: {total_properties_saved}")
    print(f"Total errors: {total_errors}")


if __name__ == "__main__":
    # Choose one of the following execution methods:

    # Method 1: Run with a single sorting method
    # main_single_sorting("photo_count", max_properties_per_state=2)

    # Method 2: Run with all sorting methods (will take longer but get more diverse data)
    main_all_sorting_methods(max_properties_per_sort=3)

    # Method 3: Run with custom selection of sorting methods
    # custom_sorts = ["newest", "lowest_price", "highest_price", "price_reduced"]
    # main_custom_sorting(custom_sorts, max_properties_per_sort=75)

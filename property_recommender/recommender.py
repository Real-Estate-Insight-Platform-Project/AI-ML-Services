import requests
import json
from supabase import create_client, Client
from datetime import datetime
import os
from typing import Dict, Any, Optional, List
import time

# Supabase configuration
SUPABASE_URL = "https://kjyiuoqbkzmbdkrpwqcr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtqeWl1b3Fia3ptYmRrcnB3cWNyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzYxNDc5OCwiZXhwIjoyMDczMTkwNzk4fQ.8t60wee7Op3EU19phk2OoxQGD-vZ1Lzxau6c50dxDj0"

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


def fetch_property_data(state_abbr: str, state_name: str, offset: int = 0):
    """Fetch property data from the Realtor API for a specific state"""
    url = "https://us-realtor.p.rapidapi.com/api/v1/property/list"

    querystring = {
        "location": state_name,
        "limit": "50",  # Increased limit to get more properties per request
        "offset": str(offset),
        "state_code": state_abbr,
        "area_type": "state",
        "sort_by": "photo_count"
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
        print(f"Error fetching data for {state_name}: {e}")
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
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
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
        return

    success_count = 0
    error_count = 0

    for property_data in properties:
        try:
            # Check if property already exists (based on address and city)
            existing_property = supabase.table('properties').select('id').eq(
                'address', property_data['address']).eq('city', property_data['city']).execute()

            if existing_property.data:
                print(f"Property already exists: {property_data['title']}")
                continue

            # Insert into Supabase if it doesn't exist
            response = supabase.table('properties').insert(
                property_data).execute()

            if response.data:
                success_count += 1
                print(f"Successfully saved property: {property_data['title']}")
            else:
                error_count += 1
                print(f"Failed to save property: {property_data['title']}")

        except Exception as e:
            error_count += 1
            print(f"Error saving property {property_data['title']}: {e}")

    print(
        f"Summary: {success_count} properties saved successfully, {error_count} failed")
    return success_count, error_count


def process_state(state: dict, max_properties: int = 200):
    """Process properties for a single state"""
    state_name = state["name"]
    state_abbr = state["abbr"]

    print(f"\nProcessing state: {state_name} ({state_abbr})")

    total_saved = 0
    total_errors = 0
    offset = 0

    while total_saved < max_properties:
        print(f"Fetching properties with offset {offset}...")

        # Fetch data from API
        api_data = fetch_property_data(state_abbr, state_name, offset)

        if not api_data:
            print(f"No data returned for {state_name}")
            break

        # Transform data
        properties = transform_property_data(api_data)

        if not properties:
            print(f"No properties found for {state_name} at offset {offset}")
            break

        print(f"Found {len(properties)} properties for {state_name}")

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
        f"Finished processing {state_name}. Total saved: {total_saved}, Total errors: {total_errors}")
    return total_saved, total_errors


def main():
    """Main function to orchestrate the process for all states"""
    total_properties_saved = 0
    total_errors = 0

    for state in US_STATES:
        saved, errors = process_state(state)
        total_properties_saved += saved
        total_errors += errors

        # Add a longer delay between states to avoid API rate limits
        time.sleep(2)

    print(f"\nFinal Summary:")
    print(
        f"Total properties saved across all states: {total_properties_saved}")
    print(f"Total errors: {total_errors}")


if __name__ == "__main__":
    main()

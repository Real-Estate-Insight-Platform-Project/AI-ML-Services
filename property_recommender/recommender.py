import requests
import json
from supabase import create_client, Client
from datetime import datetime
import os
from typing import Dict, Any, Optional

# Supabase configuration
SUPABASE_URL = "https://kjyiuoqbkzmbdkrpwqcr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtqeWl1b3Fia3ptYmRrcnB3cWNyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzYxNDc5OCwiZXhwIjoyMDczMTkwNzk4fQ.8t60wee7Op3EU19phk2OoxQGD-vZ1Lzxau6c50dxDj0"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_property_data():
    """Fetch property data from the Realtor API"""
    url = "https://us-realtor.p.rapidapi.com/api/v1/property/list"

    querystring = {
        "location": "New Hampshire",
        "limit": "10",
        "state_code": "NH",
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
        print(f"Error fetching data: {e}")
        return None


def transform_property_data(api_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Transform API data to match database schema"""
    if not api_data or 'data' not in api_data:
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
            # Insert into Supabase
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
        f"\nSummary: {success_count} properties saved successfully, {error_count} failed")


def main():
    """Main function to orchestrate the process"""
    print("Fetching property data from API...")
    api_data = fetch_property_data()

    if not api_data:
        print("Failed to fetch data from API")
        return

    print("Transforming data...")
    properties = transform_property_data(api_data)

    if not properties:
        print("No properties to process")
        return

    print(f"Processing {len(properties)} properties...")
    save_to_database(properties)


if __name__ == "__main__":
    main()

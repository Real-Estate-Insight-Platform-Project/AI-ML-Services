import os
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class SupabaseClient:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }

    def _build_query_params(self, filters: Dict[str, Any]) -> str:
        """Build query parameters from filters"""
        params = []

        # Always filter by active listings
        params.append("listing_status=eq.active")

        if filters:
            if filters.get("city"):
                params.append(f"city=ilike.%{filters['city']}%")
            if filters.get("property_type") and filters["property_type"] != "any":
                params.append(f"property_type=eq.{filters['property_type']}")
            if filters.get("min_price"):
                params.append(f"price=gte.{filters['min_price']}")
            if filters.get("max_price"):
                params.append(f"price=lte.{filters['max_price']}")
            if filters.get("min_bedrooms") and filters["min_bedrooms"] != "any":
                params.append(f"bedrooms=gte.{filters['min_bedrooms']}")
            if filters.get("min_bathrooms") and filters["min_bathrooms"] != "any":
                params.append(f"bathrooms=gte.{filters['min_bathrooms']}")

        return "&".join(params)

    def get_properties(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get properties from Supabase with optional filters"""
        try:
            query_params = self._build_query_params(filters)
            url = f"{self.url}/rest/v1/properties?{query_params}&order=created_at.desc"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            print(f"Error fetching properties: {e}")
            return []

    def get_property_by_id(self, property_id: str) -> Optional[Dict[str, Any]]:
        """Get specific property by ID"""
        try:
            url = f"{self.url}/rest/v1/properties?id=eq.{property_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            data = response.json()
            return data[0] if data else None
        except Exception as e:
            print(f"Error fetching property {property_id}: {e}")
            return None

    def get_properties_by_ids(self, property_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple properties by their IDs"""
        if not property_ids:
            return []

        try:
            # Supabase uses PostgREST syntax for IN queries
            ids_string = ",".join([f'"{id}"' for id in property_ids])
            url = f"{self.url}/rest/v1/properties?id=in.({ids_string})"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            print(f"Error fetching properties by IDs: {e}")
            return []

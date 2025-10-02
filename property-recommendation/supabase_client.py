import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


class SupabaseClient:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.client: Client = create_client(self.url, self.key)

    def get_properties(self, filters: dict = None):
        """Get properties from Supabase with optional filters"""
        query = self.client.table("properties").select(
            "*").eq("listing_status", "active")

        if filters:
            if filters.get("city"):
                query = query.ilike("city", f"%{filters['city']}%")
            if filters.get("property_type") and filters["property_type"] != "any":
                query = query.eq("property_type", filters["property_type"])
            if filters.get("min_price"):
                query = query.gte("price", int(filters["min_price"]))
            if filters.get("max_price"):
                query = query.lte("price", int(filters["max_price"]))
            if filters.get("min_bedrooms") and filters["min_bedrooms"] != "any":
                query = query.gte("bedrooms", int(filters["min_bedrooms"]))
            if filters.get("min_bathrooms") and filters["min_bathrooms"] != "any":
                query = query.gte("bathrooms", float(filters["min_bathrooms"]))

        response = query.execute()
        return response.data

    def get_property_by_id(self, property_id: str):
        """Get specific property by ID"""
        response = self.client.table("properties").select(
            "*").eq("id", property_id).execute()
        return response.data[0] if response.data else None

    def get_properties_by_ids(self, property_ids: list):
        """Get multiple properties by their IDs"""
        if not property_ids:
            return []
        response = self.client.table("properties").select(
            "*").in_("id", property_ids).execute()
        return response.data

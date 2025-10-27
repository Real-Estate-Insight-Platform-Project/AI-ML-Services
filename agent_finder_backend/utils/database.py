from supabase import create_client, Client
from config.settings import settings
from typing import Optional, List, Dict
import pandas as pd
from postgrest.exceptions import APIError


class DatabaseClient:
    """Supabase database client for agent finder."""
    
    def __init__(self):
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )
    
    def get_all_agents(self) -> pd.DataFrame:
        """Fetch all agents from database with pagination to get all records."""
        all_agents = []
        page_size = 1000
        offset = 0
        
        while True:
            response = self.client.table('real_estate_agents').select('*').range(offset, offset + page_size - 1).execute()
            
            if not response.data:
                break
                
            all_agents.extend(response.data)
            
            # If we got less than page_size records, we've reached the end
            if len(response.data) < page_size:
                break
                
            offset += page_size
            print(f"  - Loaded {len(all_agents)} agents so far...")
        
        print(f"  - Total agents loaded: {len(all_agents)}")
        return pd.DataFrame(all_agents)
    
    def get_agent_by_id(self, advertiser_id: int) -> dict:
        """Fetch single agent by ID."""
        response = self.client.table('real_estate_agents').select('*').eq(
            'advertiser_id', advertiser_id
        ).execute()
        return response.data[0] if response.data else None
    
    def get_recent_reviews_by_agent(self, advertiser_id: int, limit: int = 5) -> list:
        """Fetch recent reviews for a specific agent."""
        response = self.client.table('reviews').select('*').eq(
            'advertiser_id', advertiser_id
        ).order('review_created_date', desc=True).limit(limit).execute()
        return response.data if response.data else []
    
    def get_cities_by_state(self, state_name: str) -> list:
        """Fetch all cities for a specific state from uszips table with pagination."""
        all_cities = set()
        page_size = 1000
        offset = 0
        
        while True:
            response = self.client.table('uszips').select('city, state_name').eq(
                'state_name', state_name
            ).range(offset, offset + page_size - 1).execute()
            
            if not response.data:
                break
                
            # Add unique cities to the set
            for row in response.data:
                if row.get('city'):
                    all_cities.add(row['city'])
            
            # If we got less than page_size records, we've reached the end
            if len(response.data) < page_size:
                break
                
            offset += page_size
        
        # Convert to sorted list
        cities = list(all_cities)
        cities.sort()
        return cities
    
    def get_all_states(self) -> list:
        """Fetch all unique states from uszips table with pagination."""
        all_states = set()
        page_size = 1000
        offset = 0
        
        while True:
            response = self.client.table('uszips').select('state_name').range(offset, offset + page_size - 1).execute()
            
            if not response.data:
                break
                
            # Add unique states to the set
            for row in response.data:
                if row.get('state_name'):
                    all_states.add(row['state_name'])
            
            # If we got less than page_size records, we've reached the end
            if len(response.data) < page_size:
                break
                
            offset += page_size
        
        # Convert to sorted list
        states = list(all_states)
        states.sort()
        return states

    def get_reviews_by_agent(self, advertiser_id: int) -> pd.DataFrame:
        """Fetch all reviews for a specific agent."""
        response = self.client.table('reviews').select('*').eq(
            'advertiser_id', advertiser_id
        ).execute()
        return pd.DataFrame(response.data)
    
    def get_all_reviews(self) -> pd.DataFrame:
        """Fetch all reviews from database with pagination to get all records."""
        all_reviews = []
        page_size = 1000
        offset = 0
        
        while True:
            response = self.client.table('reviews').select('*').range(offset, offset + page_size - 1).execute()
            
            if not response.data:
                break
                
            all_reviews.extend(response.data)
            
            # If we got less than page_size records, we've reached the end
            if len(response.data) < page_size:
                break
                
            offset += page_size
            print(f"  - Loaded {len(all_reviews)} reviews so far...")
        
        print(f"  - Total reviews loaded: {len(all_reviews)}")
        return pd.DataFrame(all_reviews)
    
    def get_zipcodes(self) -> pd.DataFrame:
        """Fetch all US zipcodes with pagination to get all records."""
        all_zipcodes = []
        page_size = 1000
        offset = 0
        
        while True:
            response = self.client.table('uszips').select('*').range(offset, offset + page_size - 1).execute()
            
            if not response.data:
                break
                
            all_zipcodes.extend(response.data)
            
            # If we got less than page_size records, we've reached the end
            if len(response.data) < page_size:
                break
                
            offset += page_size
            print(f"  - Loaded {len(all_zipcodes)} zipcodes so far...")
        
        print(f"  - Total zipcodes loaded: {len(all_zipcodes)}")
        return pd.DataFrame(all_zipcodes)
    
    def update_agent(self, advertiser_id: int, data: dict):
        """Update agent data."""
        response = self.client.table('real_estate_agents').update(data).eq(
            'advertiser_id', advertiser_id
        ).execute()
        return response
    
    def update_review(self, review_id: str, data: dict):
        """Update review data."""
        response = self.client.table('reviews').update(data).eq(
            'review_id', review_id
        ).execute()
        return response
    
    def batch_update_agents(self, updates: list):
        """Batch update multiple agents."""
        for update in updates:
            advertiser_id = update.pop('advertiser_id')
            self.update_agent(advertiser_id, update)
    
    def batch_update_reviews(self, updates: list):
        """Batch update multiple reviews."""
        for update in updates:
            review_id = update.pop('review_id')
            self.update_review(review_id, update)
    
    def get_agents_by_state(self, state: str) -> pd.DataFrame:
        """Fetch agents by state."""
        response = self.client.table('real_estate_agents').select('*').eq(
            'state', state
        ).execute()
        return pd.DataFrame(response.data)
    
    def get_agents_by_zipcodes(self, zipcodes: list) -> pd.DataFrame:
        """Fetch agents serving specific zipcodes."""
        agents_df = self.get_all_agents()
        
        # Filter agents whose service_zipcodes contains any of the target zipcodes
        mask = agents_df['service_zipcodes'].apply(
            lambda x: any(z in str(x) for z in zipcodes) if pd.notna(x) else False
        )
        
        return agents_df[mask]
    
    def safe_update_review(self, review_id: str, data: dict):
        """Update review data directly."""
        if not data:
            return None
            
        try:
            response = self.client.table('reviews').update(data).eq(
                'review_id', review_id
            ).execute()
            return response
        except Exception as e:
            print(f"  - Error updating review {review_id}: {e}")
            return None
    
    def safe_update_agent(self, advertiser_id: int, data: dict):
        """Update agent data directly."""
        # Filter out the ID from update data
        update_data = {k: v for k, v in data.items() if k != 'advertiser_id'}
        
        if not update_data:
            return None
        
        # Clean data types for PostgreSQL
        cleaned_data = self._clean_data_types(update_data)
            
        try:
            response = self.client.table('real_estate_agents').update(cleaned_data).eq(
                'advertiser_id', advertiser_id
            ).execute()
            return response
        except Exception as e:
            print(f"  - Error updating agent {advertiser_id}: {e}")
            return None
    
    def batch_update_reviews_safe(self, updates: list):
        """Batch update multiple reviews."""
        for update in updates:
            review_id = update.pop('review_id', None)
            if review_id and update:
                self.safe_update_review(review_id, update)
    
    def batch_update_agents_safe(self, updates: list):
        """Batch update multiple agents."""
        for update in updates:
            advertiser_id = update.pop('advertiser_id', None)
            if advertiser_id and update:
                self.safe_update_agent(advertiser_id, update)
        # for column in missing_columns:
        #     print(f"  - Creating missing column: {column}")
        #     try:
        #         self.create_column('reviews', column, self._get_column_type(column))
        #     except Exception as e:
        #         print(f"  - Warning: Could not create column {column}: {e}")
        
        # Process updates with progress tracking
        successful_updates = 0
        skipped_updates = 0
        
        for update in updates:
            review_id = update.get('review_id')
            if not review_id:
                skipped_updates += 1
                continue
                
            try:
                self.safe_update_review(review_id, update)
                successful_updates += 1
                if successful_updates % 500 == 0:
                    print(f"  - Progress: {successful_updates} reviews updated...")
            except Exception as e:
                print(f"  - Failed to update review {review_id}: {e}")
                skipped_updates += 1
        
        print(f"  - Batch complete: {successful_updates} updated, {skipped_updates} skipped")
    
    def batch_update_agents_safe(self, updates: list):
        """Safely batch update multiple agents."""
        for update in updates:
            advertiser_id = update.get('advertiser_id')
            if advertiser_id:
                self.safe_update_agent(advertiser_id, update)

    def _clean_data_types(self, data: dict) -> dict:
        """Clean data types for PostgreSQL compatibility."""
        import pandas as pd
        import numpy as np
        
        cleaned = {}
        for key, value in data.items():
            # Handle arrays and lists (for property_types, etc.)
            if isinstance(value, (list, np.ndarray)):
                cleaned[key] = list(value) if value is not None else None
            # Handle scalar values that might be NaN
            elif hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
                # This is likely an array-like object, convert to list
                try:
                    cleaned[key] = list(value) if value is not None else None
                except:
                    cleaned[key] = value
            # Safe NaN check for scalar values only
            else:
                try:
                    is_nan = pd.isna(value)
                except (ValueError, TypeError):
                    # If pd.isna fails (e.g., with arrays), assume not NaN
                    is_nan = False
                
                if is_nan or value is None:
                    cleaned[key] = None
                elif isinstance(value, (np.floating, float)):
                    # Convert floats to integers for count/integer columns
                    if any(x in key.lower() for x in ['count', 'active_listings', 'recently_sold', 'days_since', 'experience_years', 'sub_responsiveness_count', 'sub_negotiation_count', 'sub_professionalism_count', 'sub_market_expertise_count']):
                        cleaned[key] = int(round(value)) if not is_nan else None
                    # Convert floats to integers for price columns (bigint)
                    elif any(x in key.lower() for x in ['price', 'min_price', 'max_price']):
                        cleaned[key] = int(round(value)) if not is_nan else None
                    else:
                        # Keep as float for scores, ratings, etc.
                        cleaned[key] = float(value) if not is_nan else None
                elif isinstance(value, (np.integer, int)):
                    cleaned[key] = int(value)
                elif isinstance(value, str):
                    cleaned[key] = value
                else:
                    # For any other type, try to preserve as-is
                    cleaned[key] = value
        
        return cleaned


# Global database client instance
db_client = DatabaseClient()
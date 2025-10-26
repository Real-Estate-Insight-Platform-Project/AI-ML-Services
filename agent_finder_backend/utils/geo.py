import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from geopy.distance import geodesic


class GeoUtils:
    """Utility functions for geographic calculations."""
    
    def __init__(self, uszips_df: pd.DataFrame):
        """
        Initialize with US zipcodes data.
        
        Args:
            uszips_df: DataFrame with zipcode data
        """
        self.uszips_df = uszips_df
        
        # Create zipcode lookup dictionary
        self.zipcode_coords = {}
        for _, row in uszips_df.iterrows():
            if pd.notna(row['zip']) and pd.notna(row['lat']) and pd.notna(row['lng']):
                self.zipcode_coords[str(row['zip'])] = (
                    float(row['lat']),
                    float(row['lng'])
                )
    
    def get_zipcodes_for_city(self, city: str, state: str) -> List[str]:
        """
        Get all zipcodes for a given city and state.
        
        Args:
            city: City name
            state: State abbreviation or name
        
        Returns:
            List of zipcodes
        """
        city_lower = city.lower()
        state_lower = state.lower()
        
        # Filter by city and state
        mask = (
            (self.uszips_df['city'].str.lower() == city_lower) &
            (
                (self.uszips_df['state_id'].str.lower() == state_lower) |
                (self.uszips_df['state_name'].str.lower() == state_lower)
            )
        )
        
        zipcodes = self.uszips_df[mask]['zip'].tolist()
        return [str(z) for z in zipcodes if pd.notna(z)]
    
    def get_coordinates(self, zipcode: str) -> Tuple[float, float]:
        """
        Get latitude and longitude for a zipcode.
        
        Args:
            zipcode: Zipcode string
        
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        return self.zipcode_coords.get(str(zipcode))
    
    def calculate_distance(
        self, 
        zipcode1: str, 
        zipcode2: str
    ) -> float:
        """
        Calculate distance between two zipcodes in kilometers.
        
        Args:
            zipcode1: First zipcode
            zipcode2: Second zipcode
        
        Returns:
            Distance in kilometers, or infinity if coords not found
        """
        coords1 = self.get_coordinates(zipcode1)
        coords2 = self.get_coordinates(zipcode2)
        
        if coords1 is None or coords2 is None:
            return float('inf')
        
        return geodesic(coords1, coords2).kilometers
    
    def find_nearest_zipcodes(
        self,
        target_zipcode: str,
        candidate_zipcodes: List[str],
        max_distance_km: float = None
    ) -> List[Tuple[str, float]]:
        """
        Find nearest zipcodes from candidates to target.
        
        Args:
            target_zipcode: Target zipcode
            candidate_zipcodes: List of candidate zipcodes
            max_distance_km: Maximum distance to consider (optional)
        
        Returns:
            List of (zipcode, distance) tuples, sorted by distance
        """
        if max_distance_km is None:
            max_distance_km = float('inf')
        
        distances = []
        target_coords = self.get_coordinates(target_zipcode)
        
        if target_coords is None:
            return []
        
        for zipcode in candidate_zipcodes:
            coords = self.get_coordinates(zipcode)
            if coords is not None:
                dist = geodesic(target_coords, coords).kilometers
                if dist <= max_distance_km:
                    distances.append((zipcode, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        return distances
    
    def get_nearest_service_zipcodes(
        self,
        user_zipcodes: List[str],
        all_zipcodes: List[str],
        max_distance_km: float = None
    ) -> List[str]:
        """
        Get zipcodes from all_zipcodes that are near any user_zipcode.
        
        Args:
            user_zipcodes: User's target zipcodes (from city)
            all_zipcodes: All available zipcodes to check
            max_distance_km: Maximum distance to consider
        
        Returns:
            List of nearby zipcodes
        """
        if max_distance_km is None:
            max_distance_km = float('inf')
        
        nearby_zipcodes = set()
        
        for user_zip in user_zipcodes:
            nearest = self.find_nearest_zipcodes(
                user_zip,
                all_zipcodes,
                max_distance_km
            )
            for zipcode, _ in nearest:
                nearby_zipcodes.add(zipcode)
        
        return list(nearby_zipcodes)
    
    def calculate_agent_proximity_score(
        self,
        user_zipcodes: List[str],
        agent_base_zipcode: str,
        agent_service_zipcodes: List[str]
    ) -> Dict[str, float]:
        """
        Calculate proximity score for an agent.
        
        Prioritizes:
        1. Agent base zipcode matches user zipcode
        2. Agent service zipcodes near user zipcodes
        
        Args:
            user_zipcodes: User's target zipcodes
            agent_base_zipcode: Agent's base zipcode
            agent_service_zipcodes: Agent's service zipcodes
        
        Returns:
            Dict with 'distance_km', 'proximity_score', and 'match_type'
        """
        min_distance = float('inf')
        match_type = 'none'
        
        # Check base zipcode
        if agent_base_zipcode:
            for user_zip in user_zipcodes:
                dist = self.calculate_distance(user_zip, agent_base_zipcode)
                if dist < min_distance:
                    min_distance = dist
                    match_type = 'base_zipcode'
        
        # Check service zipcodes
        if agent_service_zipcodes:
            for user_zip in user_zipcodes:
                for service_zip in agent_service_zipcodes:
                    dist = self.calculate_distance(user_zip, service_zip)
                    if dist < min_distance:
                        min_distance = dist
                        match_type = 'service_zipcode'
        
        # Calculate proximity score (exponential decay)
        if min_distance == float('inf'):
            proximity_score = 0.0
        else:
            from utils.stats import calculate_distance_score
            proximity_score = calculate_distance_score(min_distance)
        
        return {
            'distance_km': min_distance if min_distance != float('inf') else None,
            'proximity_score': proximity_score,
            'match_type': match_type
        }


# Will be initialized in the main preprocessing script
geo_utils = None
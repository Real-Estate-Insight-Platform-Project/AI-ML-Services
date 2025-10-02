import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import math


class PropertyRecommendationEngine:
    def __init__(self):
        self.scaler = StandardScaler()

    def calculate_similarity_score(self, target_property: Dict[str, Any], candidate_property: Dict[str, Any]) -> float:
        """Calculate similarity score between two properties"""

        # Price similarity (20% weight)
        price_diff = abs(
            target_property['price'] - candidate_property['price'])
        price_similarity = 1 - \
            min(price_diff / max(target_property['price'], 1), 1)

        # Bedrooms similarity (15% weight)
        bedroom_diff = abs(
            target_property['bedrooms'] - candidate_property['bedrooms'])
        bedroom_similarity = 1 - \
            (bedroom_diff / max(target_property['bedrooms'], 1))

        # Bathrooms similarity (15% weight)
        bathroom_diff = abs(
            target_property['bathrooms'] - candidate_property['bathrooms'])
        bathroom_similarity = 1 - \
            (bathroom_diff / max(target_property['bathrooms'], 1))

        # Square footage similarity (15% weight)
        sqft_diff = abs(
            target_property['square_feet'] - candidate_property['square_feet'])
        sqft_similarity = 1 - \
            (sqft_diff / max(target_property['square_feet'], 1))

        # Property type similarity (10% weight)
        type_similarity = 1 if target_property['property_type'] == candidate_property['property_type'] else 0

        # Location similarity (25% weight) - using city/state for simplicity
        # In production, you'd use actual coordinates with haversine distance
        location_similarity = 1 if (
            target_property['city'] == candidate_property['city'] and
            target_property['state'] == candidate_property['state']
        ) else 0.5 if target_property['state'] == candidate_property['state'] else 0

        # Calculate weighted score
        total_score = (
            price_similarity * 0.20 +
            bedroom_similarity * 0.15 +
            bathroom_similarity * 0.15 +
            sqft_similarity * 0.15 +
            type_similarity * 0.10 +
            location_similarity * 0.25
        )

        return total_score

    def recommend_similar_properties(
        self,
        target_property: Dict[str, Any],
        all_properties: List[Dict[str, Any]],
        filters: Dict[str, Any] = None,
        limit: int = 6
    ) -> List[Dict[str, Any]]:
        """Get similar properties based on the target property and optional filters"""

        # Filter out the target property itself
        candidate_properties = [
            prop for prop in all_properties
            if prop['id'] != target_property['id']
        ]

        # Apply additional filters if provided
        if filters:
            candidate_properties = self._apply_filters(
                candidate_properties, filters)

        # Calculate similarity scores for all candidate properties
        scored_properties = []
        for prop in candidate_properties:
            score = self.calculate_similarity_score(target_property, prop)
            scored_properties.append({
                **prop,
                'similarity_score': score
            })

        # Sort by similarity score (descending) and return top N
        scored_properties.sort(
            key=lambda x: x['similarity_score'], reverse=True)

        return scored_properties[:limit]

    def _apply_filters(self, properties: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to properties list"""
        filtered = properties

        if filters.get('city'):
            filtered = [p for p in filtered if filters['city'].lower()
                        in p['city'].lower()]

        if filters.get('property_type') and filters['property_type'] != 'any':
            filtered = [p for p in filtered if p['property_type']
                        == filters['property_type']]

        if filters.get('min_price'):
            filtered = [p for p in filtered if p['price']
                        >= int(filters['min_price'])]

        if filters.get('max_price'):
            filtered = [p for p in filtered if p['price']
                        <= int(filters['max_price'])]

        if filters.get('min_bedrooms') and filters['min_bedrooms'] != 'any':
            filtered = [p for p in filtered if p['bedrooms']
                        >= int(filters['min_bedrooms'])]

        if filters.get('min_bathrooms') and filters['min_bathrooms'] != 'any':
            filtered = [p for p in filtered if p['bathrooms']
                        >= float(filters['min_bathrooms'])]

        return filtered

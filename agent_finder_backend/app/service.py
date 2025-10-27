import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from app.models import AgentSearchRequest, AgentRecommendation
from utils.database import db_client
from utils.geo import GeoUtils
from models.scoring import agent_scorer
from config.settings import settings


class AgentRecommendationService:
    """Service for finding and ranking agent recommendations."""
    
    def __init__(self):
        """Initialize service."""
        self.geo_utils = None
    
    def initialize_geo_utils(self):
        """Initialize geographic utilities (lazy loading)."""
        if self.geo_utils is None:
            uszips_df = db_client.get_zipcodes()
            self.geo_utils = GeoUtils(uszips_df)
    
    def filter_by_state_and_city(
        self,
        agents_df: pd.DataFrame,
        state: str,
        city: str
    ) -> pd.DataFrame:
        """
        Filter agents by state and city (via service zipcodes).
        
        Args:
            agents_df: All agents
            state: State code or name
            city: City name
        
        Returns:
            Filtered agents dataframe
        """
        # Filter by state first (exact match)
        state_filtered = agents_df[
            agents_df['state'].str.upper() == state.upper()
        ].copy()
        
        if len(state_filtered) == 0:
            return pd.DataFrame()
        
        # Get zipcodes for the city
        self.initialize_geo_utils()
        city_zipcodes = self.geo_utils.get_zipcodes_for_city(city, state)
        
        if not city_zipcodes:
            # If no zipcodes found, try to match by city name
            city_filtered = state_filtered[
                state_filtered['agent_base_city'].str.lower() == city.lower()
            ]
            return city_filtered
        
        # Filter agents who serve these zipcodes
        mask = state_filtered.apply(
            lambda row: (
                # Agent is based in one of the city's zipcodes
                (str(row['agent_base_zipcode']) in city_zipcodes) or
                # Or agent serves one of the city's zipcodes
                any(
                    zipcode in row.get('service_zipcodes_list', [])
                    for zipcode in city_zipcodes
                )
            ),
            axis=1
        )
        
        city_filtered = state_filtered[mask].copy()
        
        # If very few agents found, expand to nearby zipcodes
        if len(city_filtered) < 5:
            # Get all unique service zipcodes from agents in state
            all_service_zips = set()
            for zips in state_filtered['service_zipcodes_list']:
                all_service_zips.update(zips)
            
            # Find nearby zipcodes
            nearby_zips = self.geo_utils.get_nearest_service_zipcodes(
                city_zipcodes,
                list(all_service_zips),
                max_distance_km=50  # 50km radius
            )
            
            # Add agents serving nearby zipcodes
            expanded_mask = state_filtered.apply(
                lambda row: any(
                    zipcode in row.get('service_zipcodes_list', [])
                    for zipcode in nearby_zips
                ),
                axis=1
            )
            
            expanded_filtered = state_filtered[expanded_mask].copy()
            
            # Combine and deduplicate
            city_filtered = pd.concat([city_filtered, expanded_filtered]).drop_duplicates(
                subset='advertiser_id'
            )
        
        return city_filtered
    
    def filter_by_price_range(
        self,
        agents_df: pd.DataFrame,
        min_price: Optional[float],
        max_price: Optional[float]
    ) -> pd.DataFrame:
        """
        Filter agents by price range.
        
        Args:
            agents_df: Agents dataframe
            min_price: Minimum price
            max_price: Maximum price
        
        Returns:
            Filtered dataframe
        """
        if min_price is None and max_price is None:
            return agents_df
        
        filtered = agents_df.copy()
        
        # Agent's range should overlap with user's range
        if min_price is not None:
            # Agent's max price should be >= user's min price
            filtered = filtered[
                (filtered['max_price'].isna()) |
                (filtered['max_price'] >= min_price)
            ]
        
        if max_price is not None:
            # Agent's min price should be <= user's max price
            filtered = filtered[
                (filtered['min_price'].isna()) |
                (filtered['min_price'] <= max_price)
            ]
        
        return filtered
    
    def filter_by_property_type(
        self,
        agents_df: pd.DataFrame,
        property_type: Optional[str]
    ) -> pd.DataFrame:
        """
        Filter agents by property type specialization.
        
        Args:
            agents_df: Agents dataframe
            property_type: Property type to filter
        
        Returns:
            Filtered dataframe
        """
        if property_type is None:
            return agents_df
        
        # Check if property type is in agent's property_types list
        mask = agents_df['property_types'].apply(
            lambda types: property_type in types if isinstance(types, list) else False
        )
        
        filtered = agents_df[mask]
        
        # If too few results, return all (soft filter)
        if len(filtered) < 3:
            return agents_df
        
        return filtered
    
    def filter_by_language(
        self,
        agents_df: pd.DataFrame,
        language: str
    ) -> pd.DataFrame:
        """
        Filter agents by language.
        
        Args:
            agents_df: Agents dataframe
            language: Language preference
        
        Returns:
            Filtered dataframe
        """
        if language is None or language.lower() == 'english':
            # Most agents speak English, so no filtering needed
            return agents_df
        
        mask = agents_df['languages'].apply(
            lambda langs: language.lower() in str(langs).lower() if pd.notna(langs) else False
        )
        
        filtered = agents_df[mask]
        
        # Soft filter
        if len(filtered) < 3:
            return agents_df
        
        return filtered
    
    def apply_specialization_preference(
        self,
        agents_df: pd.DataFrame,
        specializations: List[str]
    ) -> pd.DataFrame:
        """
        Prefer agents with additional specializations (soft filter).
        
        Args:
            agents_df: Agents dataframe
            specializations: List of preferred specializations
        
        Returns:
            Dataframe with specialization_match_score added
        """
        if not specializations:
            agents_df['specialization_match_score'] = 0.5
            return agents_df
        
        # Calculate match score
        def calculate_match(agent_specs):
            if not isinstance(agent_specs, list):
                return 0.0
            
            matches = sum(1 for spec in specializations if spec in agent_specs)
            return matches / len(specializations)
        
        agents_df['specialization_match_score'] = agents_df['additional_specializations'].apply(
            calculate_match
        )
        
        return agents_df
    
    def calculate_proximity_scores(
        self,
        agents_df: pd.DataFrame,
        city: str,
        state: str
    ) -> pd.Series:
        """
        Calculate proximity scores for all agents.
        
        Args:
            agents_df: Agents dataframe
            city: User's city
            state: User's state
        
        Returns:
            Series of proximity scores
        """
        self.initialize_geo_utils()
        
        # Get user's zipcodes
        user_zipcodes = self.geo_utils.get_zipcodes_for_city(city, state)
        
        if not user_zipcodes:
            return pd.Series([0.5] * len(agents_df), index=agents_df.index)
        
        proximity_scores = []
        
        for _, agent in agents_df.iterrows():
            proximity_data = self.geo_utils.calculate_agent_proximity_score(
                user_zipcodes,
                agent.get('agent_base_zipcode'),
                agent.get('service_zipcodes_list', [])
            )
            proximity_scores.append(proximity_data['proximity_score'])
        
        return pd.Series(proximity_scores, index=agents_df.index)
    
    def find_agents(
        self,
        request: AgentSearchRequest
    ) -> List[AgentRecommendation]:
        """
        Find and rank agents based on user request.
        
        Args:
            request: Agent search request
        
        Returns:
            List of agent recommendations, sorted by matching score
        """
        # Load all agents
        agents_df = db_client.get_all_agents()
        
        if len(agents_df) == 0:
            return []
        
        # Apply filters
        print(f"Initial agents: {len(agents_df)}")
        
        # Filter by state and city
        agents_df = self.filter_by_state_and_city(
            agents_df,
            request.state,
            request.city
        )
        print(f"After state/city filter: {len(agents_df)}")
        
        if len(agents_df) == 0:
            return []
        
        # Filter by price range
        agents_df = self.filter_by_price_range(
            agents_df,
            request.min_price,
            request.max_price
        )
        print(f"After price filter: {len(agents_df)}")
        
        # Filter by property type (soft)
        agents_df = self.filter_by_property_type(
            agents_df,
            request.property_type
        )
        print(f"After property type filter: {len(agents_df)}")
        
        # Filter by language (soft)
        agents_df = self.filter_by_language(
            agents_df,
            request.language
        )
        print(f"After language filter: {len(agents_df)}")
        
        # Apply specialization preference (soft)
        agents_df = self.apply_specialization_preference(
            agents_df,
            request.additional_specializations or []
        )
        
        # Calculate proximity scores
        proximity_scores = self.calculate_proximity_scores(
            agents_df,
            request.city,
            request.state
        )
        
        # Prepare user preferences (normalize weights)
        sub_score_prefs = request.sub_score_preferences or {}
        if sub_score_prefs:
            total = sum(sub_score_prefs.values())
            if total > 0:
                sub_score_prefs = {k: v/total for k, v in sub_score_prefs.items()}
        else:
            # Default equal weights
            sub_score_prefs = {
                'responsiveness': 0.25,
                'negotiation': 0.25,
                'professionalism': 0.25,
                'market_expertise': 0.25
            }
        
        skill_prefs = request.skill_preferences or {}
        if skill_prefs:
            total = sum(skill_prefs.values())
            if total > 0:
                skill_prefs = {k: v/total for k, v in skill_prefs.items()}
        
        # Score agents
        agents_scored = agent_scorer.score_agents(
            agents_df,
            request.user_type,
            sub_score_prefs,
            skill_prefs,
            request.is_urgent,
            proximity_scores
        )
        
        # Boost score for specialization matches
        if 'specialization_match_score' in agents_scored.columns:
            agents_scored['matching_score'] = (
                0.9 * agents_scored['matching_score'] +
                0.1 * agents_scored['specialization_match_score']
            )
            agents_scored['matching_score_display'] = (
                agents_scored['matching_score'] * 100
            ).round(1)
        
        # Sort by matching score
        agents_scored = agents_scored.sort_values(
            'matching_score',
            ascending=False
        )
        
        # Limit results
        agents_scored = agents_scored.head(request.max_results)
        
        # Convert to recommendations
        recommendations = []
        
        for _, agent in agents_scored.iterrows():
            # Determine buyer/seller fit
            buyer_sat = agent.get('buyer_satisfaction', 0.5)
            seller_sat = agent.get('seller_satisfaction', 0.5)
            
            if buyer_sat > 0.7 and seller_sat > 0.7:
                fit = 'both'
            elif buyer_sat > seller_sat:
                fit = 'buyer'
            else:
                fit = 'seller'
            
            # Get distance
            self.initialize_geo_utils()
            user_zipcodes = self.geo_utils.get_zipcodes_for_city(
                request.city,
                request.state
            )
            
            if user_zipcodes and agent.get('agent_base_zipcode'):
                distance_km = self.geo_utils.calculate_distance(
                    user_zipcodes[0],
                    agent['agent_base_zipcode']
                )
                if distance_km == float('inf'):
                    distance_km = None
            else:
                distance_km = None
            
            recommendation = AgentRecommendation(
                advertiser_id=int(agent['advertiser_id']),
                full_name=agent['full_name'],
                state=agent['state'],
                agent_base_city=agent['agent_base_city'],
                agent_base_zipcode=agent.get('agent_base_zipcode'),
                phone_primary=agent.get('phone_primary'),
                office_phone=agent.get('office_phone'),
                agent_website=agent.get('agent_website'),
                office_name=agent.get('office_name'),
                has_photo=bool(agent.get('has_photo', False)),
                agent_photo_url=agent.get('agent_photo_url'),
                experience_years=float(agent['experience_years']) if pd.notna(agent.get('experience_years')) else None,
                matching_score=float(agent['matching_score_display']),
                proximity_score=float(agent['proximity_score']),
                distance_km=distance_km,
                review_count=int(agent.get('review_count', 0)),
                agent_rating=float(agent.get('agent_rating', 0)) if pd.notna(agent.get('agent_rating')) else 0.0,
                positive_review_count=int(agent.get('positive_review_count', 0)),
                negative_review_count=int(agent.get('negative_review_count', 0)),
                recently_sold_count=int(agent.get('recently_sold_count', 0)),
                active_listings_count=int(agent.get('active_listings_count', 0)),
                days_since_last_sale=int(agent['days_since_last_sale']) if pd.notna(agent.get('days_since_last_sale')) else None,
                property_types=agent.get('property_types', []) if isinstance(agent.get('property_types'), list) else [],
                additional_specializations=agent.get('additional_specializations', []) if isinstance(agent.get('additional_specializations'), list) else [],
                avg_responsiveness=float(agent['avg_sub_responsiveness']) if pd.notna(agent.get('avg_sub_responsiveness')) else None,
                avg_negotiation=float(agent['avg_sub_negotiation']) if pd.notna(agent.get('avg_sub_negotiation')) else None,
                avg_professionalism=float(agent['avg_sub_professionalism']) if pd.notna(agent.get('avg_sub_professionalism')) else None,
                avg_market_expertise=float(agent['avg_sub_market_expertise']) if pd.notna(agent.get('avg_sub_market_expertise')) else None,
                buyer_seller_fit=fit
            )
            
            recommendations.append(recommendation)
        
        return recommendations


# Global service instance
recommendation_service = AgentRecommendationService()
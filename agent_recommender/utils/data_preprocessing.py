"""
Data preprocessing utilities for the Agent Recommender System.

This module provides functions to load and preprocess agent data from statewise JSON files,
extract features, and prepare data for ranking algorithms.
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentDataLoader:
    """Load and preprocess agent data from statewise JSON files."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the directory containing statewise JSON files
        """
        self.data_path = Path(data_path)
        self.agents_df = None
        
    def load_all_agents(self) -> pd.DataFrame:
        """
        Load all agents from all state files and combine into a DataFrame.
        
        Returns:
            pandas.DataFrame: Combined agent data
        """
        all_agents = []
        
        # Get all JSON files in the statewise_data directory
        json_files = list(self.data_path.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.data_path}")
        
        logger.info(f"Loading agent data from {len(json_files)} state files...")
        
        for json_file in json_files:
            state_name = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'data' in data and isinstance(data['data'], list):
                    for agent in data['data']:
                        agent['state'] = state_name
                        all_agents.append(agent)
                        
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(all_agents)} agents from all states")
        self.agents_df = pd.DataFrame(all_agents)
        return self.agents_df
    
    def preprocess_agents(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess agent data to extract useful features.
        
        Args:
            df: Raw agent DataFrame
            
        Returns:
            pandas.DataFrame: Preprocessed agent data
        """
        df = df.copy()
        
        # Handle missing values
        df['starRating'] = df['starRating'].fillna(0)
        df['numReviews'] = df['numReviews'].fillna(0)
        df['pastYearDeals'] = df['pastYearDeals'].fillna(0)
        df['pastYearDealsInRegion'] = df['pastYearDealsInRegion'].fillna(0)
        df['homeTransactionsLifetime'] = df['homeTransactionsLifetime'].fillna(0)
        df['transactionVolumeLifetime'] = df['transactionVolumeLifetime'].fillna(0)
        
        # Extract deal price statistics
        df['dealPrices_median'] = df['dealPrices'].apply(self._calculate_median_price)
        df['dealPrices_q25'] = df['dealPrices'].apply(self._calculate_q25_price)
        df['dealPrices_q75'] = df['dealPrices'].apply(self._calculate_q75_price)
        df['dealPrices_min'] = df['dealPrices'].apply(self._calculate_min_price)
        df['dealPrices_max'] = df['dealPrices'].apply(self._calculate_max_price)
        df['dealPrices_count'] = df['dealPrices'].apply(len)
        df['dealPrices_std'] = df['dealPrices'].apply(self._calculate_std_price)
        
        # Extract primary service region count
        df['num_service_regions'] = df['primaryServiceRegions'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # Extract property type count
        df['num_property_types'] = df['propertyTypes'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # Calculate experience metrics
        df['avg_transaction_value'] = np.where(
            df['homeTransactionsLifetime'] > 0,
            df['transactionVolumeLifetime'] / df['homeTransactionsLifetime'],
            0
        )
        
        # Boolean flags
        df['partner'] = df['partner'].fillna(False)
        df['isPremier'] = df['isPremier'].fillna(False)
        df['servesOffers'] = df['servesOffers'].fillna(False)
        df['servesListings'] = df['servesListings'].fillna(False)
        df['isActive'] = df['isActive'].fillna(False)
        df['profileContactEnabled'] = df['profileContactEnabled'].fillna(False)
        
        return df
    
    def _calculate_median_price(self, prices: List[float]) -> float:
        """Calculate median price from deal prices list."""
        if not prices or not isinstance(prices, list):
            return 0
        return np.median([p for p in prices if p > 0])
    
    def _calculate_q25_price(self, prices: List[float]) -> float:
        """Calculate 25th percentile price."""
        if not prices or not isinstance(prices, list):
            return 0
        return np.percentile([p for p in prices if p > 0], 25) if prices else 0
    
    def _calculate_q75_price(self, prices: List[float]) -> float:
        """Calculate 75th percentile price."""
        if not prices or not isinstance(prices, list):
            return 0
        return np.percentile([p for p in prices if p > 0], 75) if prices else 0
    
    def _calculate_min_price(self, prices: List[float]) -> float:
        """Calculate minimum price."""
        if not prices or not isinstance(prices, list):
            return 0
        return min([p for p in prices if p > 0]) if prices else 0
    
    def _calculate_max_price(self, prices: List[float]) -> float:
        """Calculate maximum price."""
        if not prices or not isinstance(prices, list):
            return 0
        return max([p for p in prices if p > 0]) if prices else 0
    
    def _calculate_std_price(self, prices: List[float]) -> float:
        """Calculate standard deviation of prices."""
        if not prices or not isinstance(prices, list):
            return 0
        return np.std([p for p in prices if p > 0]) if len(prices) > 1 else 0


class FeatureExtractor:
    """Extract features for agent ranking."""
    
    def __init__(self, agents_df: pd.DataFrame):
        """
        Initialize feature extractor with agent data.
        
        Args:
            agents_df: Preprocessed agent DataFrame
        """
        self.agents_df = agents_df
        self.market_stats = self._calculate_market_stats()
    
    def _calculate_market_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate market-level statistics for normalization."""
        stats = {}
        
        # Group by business market
        for market in self.agents_df['businessMarket'].unique():
            if pd.isna(market):
                continue
                
            market_agents = self.agents_df[
                self.agents_df['businessMarket'] == market
            ]
            
            stats[market] = {
                'pastYearDeals_min': market_agents['pastYearDeals'].min(),
                'pastYearDeals_max': market_agents['pastYearDeals'].max(),
                'starRating_mean': market_agents['starRating'].mean(),
                'numReviews_mean': market_agents['numReviews'].mean(),
                'dealPrices_median_mean': market_agents['dealPrices_median'].mean(),
            }
        
        return stats
    
    def calculate_geo_overlap(self, user_regions: List[str], agent_regions: List[str]) -> float:
        """
        Calculate geographical overlap between user target regions and agent service regions.
        
        Args:
            user_regions: List of user's target regions
            agent_regions: List of agent's primary service regions
            
        Returns:
            float: Overlap score (0-1)
        """
        if not user_regions or not agent_regions:
            return 0.0
        
        user_set = set([region.lower().strip() for region in user_regions])
        agent_set = set([region.lower().strip() for region in agent_regions])
        
        intersection = len(user_set.intersection(agent_set))
        return intersection / len(user_set) if user_set else 0.0
    
    def calculate_price_band_match(self, user_budget: float, agent_deal_prices: List[float], 
                                 delta: float = 0.15) -> float:
        """
        Calculate price band match score.
        
        Args:
            user_budget: User's budget
            agent_deal_prices: Agent's historical deal prices
            delta: Budget tolerance (±15% by default)
            
        Returns:
            float: Price band match score (0-1)
        """
        if not agent_deal_prices or user_budget <= 0:
            return 0.0
        
        budget_min = user_budget * (1 - delta)
        budget_max = user_budget * (1 + delta)
        
        matching_deals = [
            price for price in agent_deal_prices 
            if budget_min <= price <= budget_max
        ]
        
        return len(matching_deals) / len(agent_deal_prices)
    
    def calculate_property_type_match(self, user_property_types: List[str], 
                                    agent_property_types: List[str]) -> float:
        """
        Calculate Jaccard similarity for property types.
        
        Args:
            user_property_types: User's desired property types
            agent_property_types: Agent's handled property types
            
        Returns:
            float: Jaccard similarity score (0-1)
        """
        if not user_property_types or not agent_property_types:
            return 0.0
        
        user_set = set([pt.lower().strip() for pt in user_property_types])
        agent_set = set([pt.lower().strip() for pt in agent_property_types])
        
        intersection = len(user_set.intersection(agent_set))
        union = len(user_set.union(agent_set))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_normalized_recency(self, agent_past_year_deals: int, 
                                   market: str) -> float:
        """
        Calculate normalized recency score within market.
        
        Args:
            agent_past_year_deals: Agent's deals in past year
            market: Business market
            
        Returns:
            float: Normalized recency score (0-1)
        """
        if market not in self.market_stats:
            return 0.0
        
        min_deals = self.market_stats[market]['pastYearDeals_min']
        max_deals = self.market_stats[market]['pastYearDeals_max']
        
        if max_deals == min_deals:
            return 1.0 if agent_past_year_deals > 0 else 0.0
        
        return (agent_past_year_deals - min_deals) / (max_deals - min_deals)
    
    def calculate_rating_score(self, star_rating: float, num_reviews: int, 
                             prior_weight: float = 5.0) -> float:
        """
        Calculate Wilson score for star rating with reliability prior.
        
        Args:
            star_rating: Agent's star rating (1-5)
            num_reviews: Number of reviews
            prior_weight: Weight for prior belief
            
        Returns:
            float: Adjusted rating score (0-1)
        """
        if star_rating <= 0:
            return 0.0
        
        # Convert 1-5 scale to 0-1 scale
        normalized_rating = (star_rating - 1) / 4
        
        # Apply Bayesian smoothing with prior
        prior_rating = 0.8  # Assume 4.2/5 as prior (0.8 on 0-1 scale)
        
        adjusted_rating = (
            (num_reviews * normalized_rating + prior_weight * prior_rating) /
            (num_reviews + prior_weight)
        )
        
        return adjusted_rating
    
    def extract_features_for_query(self, user_query: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract all features for a user query against all agents.
        
        Args:
            user_query: Dictionary containing user preferences
                - regions: List[str] - target regions
                - budget: float - budget amount
                - property_types: List[str] - desired property types
                
        Returns:
            pandas.DataFrame: DataFrame with calculated features
        """
        features_df = self.agents_df.copy()
        
        # Calculate all features
        features_df['geo_overlap'] = features_df['primaryServiceRegions'].apply(
            lambda regions: self.calculate_geo_overlap(
                user_query.get('regions', []), regions
            )
        )
        
        features_df['price_band_match'] = features_df['dealPrices'].apply(
            lambda prices: self.calculate_price_band_match(
                user_query.get('budget', 0), prices
            )
        )
        
        features_df['property_type_match'] = features_df['propertyTypes'].apply(
            lambda types: self.calculate_property_type_match(
                user_query.get('property_types', []), types
            )
        )
        
        features_df['normalized_recency'] = features_df.apply(
            lambda row: self.calculate_normalized_recency(
                row['pastYearDeals'], row['businessMarket']
            ), axis=1
        )
        
        features_df['rating_score'] = features_df.apply(
            lambda row: self.calculate_rating_score(
                row['starRating'], row['numReviews']
            ), axis=1
        )
        
        # Partner/Premier boost
        features_df['partner_premier_boost'] = (
            features_df['partner'].astype(float) * 0.5 +
            features_df['isPremier'].astype(float) * 0.5
        )
        
        # Log of reviews (for diminishing returns)
        features_df['log_reviews'] = np.log1p(features_df['numReviews'])
        features_df['log_reviews_normalized'] = (
            features_df['log_reviews'] / features_df['log_reviews'].max()
            if features_df['log_reviews'].max() > 0 else 0
        )
        
        return features_df


if __name__ == "__main__":
    # Example usage
    data_path = "../agent_data/statewise_data"
    
    # Load and preprocess data
    loader = AgentDataLoader(data_path)
    agents_df = loader.load_all_agents()
    agents_df = loader.preprocess_agents(agents_df)
    
    print(f"Loaded {len(agents_df)} agents")
    print(f"Unique states: {agents_df['state'].nunique()}")
    print(f"Unique business markets: {agents_df['businessMarket'].nunique()}")
    
    # Example feature extraction
    feature_extractor = FeatureExtractor(agents_df)
    
    # Example user query
    user_query = {
        'regions': ['Fort Myers', 'Naples'],
        'budget': 400000,
        'property_types': ['Single Family Residential', 'Condo/Co-op']
    }
    
    features_df = feature_extractor.extract_features_for_query(user_query)
    print(f"Features extracted for {len(features_df)} agents")
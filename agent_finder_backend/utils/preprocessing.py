import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
from utils.stats import (
    wilson_lower_bound,
    bayesian_rating_shrinkage,
    recency_weighted_volume_score,
    exponential_decay_score
)
from models.sentiment import sentiment_analyzer
from models.skills import skill_extractor
from config.settings import settings


class DataPreprocessor:
    """Preprocess and engineer features for agent recommendation system."""
    
    # Standard property types
    PROPERTY_TYPES = [
        'single_family',
        'multi_family',
        'condo',
        'townhouse',
        'land',
        'commercial',
        'luxury',
        'new_construction'
    ]
    
    # Additional specializations
    ADDITIONAL_SPECIALIZATIONS = [
        'first_time_buyer',
        'investor',
        'veteran',
        'senior',
        'relocation',
        'foreclosure',
        'short_sale',
        'rental'
    ]
    
    def __init__(self):
        """Initialize preprocessor."""
        self.property_type_keywords = {
            'single_family': ['single family', 'single-family', 'sfh', 'house', 'home'],
            'multi_family': ['multi family', 'multi-family', 'duplex', 'triplex', 'fourplex'],
            'condo': ['condo', 'condominium'],
            'townhouse': ['townhouse', 'town house', 'townhome'],
            'land': ['land', 'lot', 'acreage', 'vacant'],
            'commercial': ['commercial', 'business', 'retail', 'office'],
            'luxury': ['luxury', 'high-end', 'estate'],
            'new_construction': ['new construction', 'new build', 'builder']
        }
        
        self.additional_spec_keywords = {
            'first_time_buyer': ['first time', 'first-time', 'new buyer'],
            'investor': ['investor', 'investment', 'rental property'],
            'veteran': ['veteran', 'military', 'va loan'],
            'senior': ['senior', '55+', 'retirement', 'downsizing'],
            'relocation': ['relocation', 'relocating', 'moving', 'transfer'],
            'foreclosure': ['foreclosure', 'reo', 'bank-owned'],
            'short_sale': ['short sale', 'short-sale'],
            'rental': ['rental', 'lease', 'tenant']
        }
    
    def parse_specializations(self, specialization_text: str) -> Dict[str, List[str]]:
        """
        Parse and standardize specializations into property types and additional specs.
        
        Args:
            specialization_text: Raw specialization text
        
        Returns:
            Dict with 'property_types' and 'additional_specializations' lists
        """
        if pd.isna(specialization_text):
            return {'property_types': [], 'additional_specializations': []}
        
        text_lower = str(specialization_text).lower()
        
        property_types = []
        additional_specs = []
        
        # Check property types
        for prop_type, keywords in self.property_type_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                property_types.append(prop_type)
        
        # Check additional specializations
        for spec, keywords in self.additional_spec_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                additional_specs.append(spec)
        
        return {
            'property_types': property_types,
            'additional_specializations': additional_specs
        }
    
    def preprocess_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess reviews with sentiment analysis and skill extraction.
        
        Args:
            reviews_df: Raw reviews dataframe
        
        Returns:
            Processed reviews dataframe
        """
        print("Analyzing review sentiments...")
        reviews_df = sentiment_analyzer.analyze_reviews_batch(reviews_df)
        
        # Calculate days since review
        reviews_df['review_created_date'] = pd.to_datetime(
            reviews_df['review_created_date']
        )
        current_date = datetime.now()
        reviews_df['days_since_review'] = (
            current_date - reviews_df['review_created_date']
        ).dt.days
        
        # Calculate recency weight
        reviews_df['recency_weight'] = reviews_df['days_since_review'].apply(
            lambda d: exponential_decay_score(d, settings.REVIEW_RECENCY_DECAY)
        )
        
        return reviews_df
    
    def aggregate_review_metrics(
        self, 
        reviews_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate review-level metrics to agent level.
        
        Args:
            reviews_df: Processed reviews dataframe
        
        Returns:
            DataFrame with agent-level aggregated metrics
        """
        # Group by agent
        agent_metrics = []
        
        for agent_id, group in reviews_df.groupby('advertiser_id'):
            metrics = {'advertiser_id': agent_id}
            
            # Count positive and negative reviews
            metrics['positive_review_count'] = (
                group['sentiment'] == 'positive'
            ).sum()
            metrics['negative_review_count'] = (
                group['sentiment'] == 'negative'
            ).sum()
            metrics['neutral_review_count'] = (
                group['sentiment'] == 'neutral'
            ).sum()
            
            # Calculate Wilson lower bound for review quality
            total_reviews = len(group)
            metrics['wilson_score'] = wilson_lower_bound(
                metrics['positive_review_count'],
                total_reviews
            )

            # REPLACE existing sub-score aggregation code:
            for sub_col in ['sub_responsiveness', 'sub_negotiation', 
                            'sub_professionalism', 'sub_market_expertise']:
                
                metrics[f'avg_{sub_col}'] = self._aggregate_subscore_with_missing_handling(
                    reviews_df, 
                    agent_id, 
                    sub_col
                )
            
            # Determine buyer/seller suitability
            buyer_reviews = group[group['reviewer_role'] == 'BUYER']
            seller_reviews = group[group['reviewer_role'] == 'SELLER']
            
            buyer_positive = (
                (buyer_reviews['sentiment'] == 'positive').sum()
            )
            seller_positive = (
                (seller_reviews['sentiment'] == 'positive').sum()
            )
            
            metrics['buyer_review_count'] = len(buyer_reviews)
            metrics['seller_review_count'] = len(seller_reviews)
            metrics['buyer_positive_count'] = buyer_positive
            metrics['seller_positive_count'] = seller_positive
            
            # Calculate buyer/seller scores
            if len(buyer_reviews) > 0:
                metrics['buyer_satisfaction'] = buyer_positive / len(buyer_reviews)
            else:
                metrics['buyer_satisfaction'] = 0.5  # Neutral
            
            if len(seller_reviews) > 0:
                metrics['seller_satisfaction'] = seller_positive / len(seller_reviews)
            else:
                metrics['seller_satisfaction'] = 0.5  # Neutral
            
            agent_metrics.append(metrics)
        
        return pd.DataFrame(agent_metrics)
    
    def preprocess_agents(
        self, 
        agents_df: pd.DataFrame,
        review_metrics_df: pd.DataFrame,
        skill_scores_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Preprocess agents with all feature engineering.
        
        Args:
            agents_df: Raw agents dataframe
            review_metrics_df: Aggregated review metrics
            skill_scores_df: Aggregated skill scores
        
        Returns:
            Processed agents dataframe
        """
        print("Processing agent features...")
        
        # Debug: Check initial column count
        print(f"  - Initial agents_df columns: {len(agents_df.columns)}")
        print(f"  - Review metrics columns: {list(review_metrics_df.columns)}")
        print(f"  - Review metrics shape: {review_metrics_df.shape}")
        
        # Merge review metrics
        agents_df = agents_df.merge(
            review_metrics_df,
            on='advertiser_id',
            how='left'
        )
        
        # Debug: Check after review metrics merge
        print(f"  - After review merge, columns: {len(agents_df.columns)}")
        print(f"  - Has positive_review_count: {'positive_review_count' in agents_df.columns}")
        if 'positive_review_count' in agents_df.columns:
            positive_count = agents_df['positive_review_count'].sum()
            print(f"  - Total positive reviews in merged data: {positive_count}")
        
        # Merge skill scores
        agents_df = agents_df.merge(
            skill_scores_df,
            on='advertiser_id',
            how='left'
        )
        
        # Fill NaN values for metrics
        metric_columns = [
            'positive_review_count', 'negative_review_count',
            'neutral_review_count', 'wilson_score',
            'buyer_satisfaction', 'seller_satisfaction'
        ]
        for col in metric_columns:
            if col in agents_df.columns:
                agents_df[col] = agents_df[col].fillna(0)
        
        # Apply Bayesian shrinkage to agent_rating
        agents_df['shrunk_rating'] = agents_df.apply(
            lambda row: bayesian_rating_shrinkage(
                row['agent_rating'] if pd.notna(row['agent_rating']) else 4.0,
                row['review_count'] if pd.notna(row['review_count']) else 0
            ),
            axis=1
        )
        
        # Calculate performance score (recency + volume)
        agents_df['performance_score'] = agents_df.apply(
            lambda row: recency_weighted_volume_score(
                row['recently_sold_count'] if pd.notna(row['recently_sold_count']) else 0,
                row['days_since_last_sale'] if pd.notna(row['days_since_last_sale']) else 365
            ),
            axis=1
        )
        
        # Parse specializations
        specializations = agents_df['specializations'].apply(
            self.parse_specializations
        )
        agents_df['property_types'] = specializations.apply(
            lambda x: x['property_types']
        )
        agents_df['additional_specializations'] = specializations.apply(
            lambda x: x['additional_specializations']
        )
        
        # Parse service zipcodes into list
        agents_df['service_zipcodes_list'] = agents_df['service_zipcodes'].apply(
            lambda x: [z.strip() for z in str(x).split(',')] if pd.notna(x) else []
        )
        
        
        agents_df[['min_price', 'max_price']] = agents_df.apply(
            lambda row: pd.Series(self._calculate_robust_price_range(row)),
            axis=1
        )
        
        # Handle missing experience
        agents_df['experience_years'] = agents_df['experience_years'].fillna(0)
        
        # Normalize experience score (log scale)
        agents_df['experience_score'] = agents_df['experience_years'].apply(
            lambda x: np.log1p(x) / np.log1p(30) if x > 0 else 0  # Max 30 years
        )
        
        return agents_df

    def _aggregate_subscore_with_missing_handling(
        self, 
        reviews_df: pd.DataFrame, 
        agent_id: int,
        subscore_col: str
    ) -> float:
        """
        Aggregate sub-scores with intelligent missing value handling.
        
        For agents with many valid scores:
            Use agent's own average
        For agents with few scores:
            Blend with global average  
        For agents with no scores:
            Use global average
        
        Args:
            reviews_df: Reviews dataframe
            agent_id: Agent ID
            subscore_col: Sub-score column name
        
        Returns:
            Aggregated sub-score
        """
        agent_reviews = reviews_df[reviews_df['advertiser_id'] == agent_id]
        valid_scores = agent_reviews[subscore_col].dropna()
        
        # Calculate global average (cache this in practice)
        global_avg = reviews_df[subscore_col].mean()
        
        if len(valid_scores) >= 3:
            # Enough data - use agent's average
            return float(valid_scores.mean())
        elif len(valid_scores) > 0:
            # Some data - blend with global average
            agent_avg = valid_scores.mean()
            # Weight more toward agent's data if they have 2 reviews
            weight = len(valid_scores) / 3.0
            return float(weight * agent_avg + (1 - weight) * global_avg)
        else:
            # No data - use global average
            return float(global_avg)


    def _calculate_robust_price_range(self, row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate robust price range from all available sources.
        
        Handles edge cases:
        - All prices are 0
        - Only some price fields have data
        - Outliers in the data
        
        Args:
            row: Agent row with price columns
        
        Returns:
            Tuple of (min_price, max_price) or (None, None)
        """
        prices = []
        
        # Collect all valid prices
        price_fields = [
            'active_listings_min_price',
            'active_listings_max_price',
            'recently_sold_min_price',
            'recently_sold_max_price'
        ]
        
        for field in price_fields:
            if field in row.index:
                price = row[field]
                if pd.notna(price) and price > 0:
                    prices.append(float(price))
        
        if len(prices) == 0:
            # Agent has no price history
            return None, None
        
        if len(prices) == 1:
            # Only one price point
            return prices[0], prices[0]
        
        # Use percentiles to handle outliers
        # 10th percentile for min, 90th for max
        prices_arr = np.array(prices)
        min_price = np.percentile(prices_arr, 10)
        max_price = np.percentile(prices_arr, 90)
        
        return float(min_price), float(max_price)


# Global preprocessor instance
preprocessor = DataPreprocessor()
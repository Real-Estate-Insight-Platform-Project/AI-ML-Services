"""
Data processing and ETL pipeline for agent finder system.

Handles:
- Loading and cleaning agent and review data
- Date parsing and recency calculations
- Data validation and missing value handling
- Agent-review joining and aggregation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import logging
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processing class for agent finder system."""
    
    def __init__(self, agents_file: str, reviews_file: str):
        """Initialize with file paths."""
        self.agents_file = agents_file
        self.reviews_file = reviews_file
        self.agents_df = None
        self.reviews_df = None
        self.current_date = datetime.now()
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and perform initial processing of agent and review data."""
        logger.info("Loading agent and review data...")
        
        # Load datasets
        self.agents_df = pd.read_csv(self.agents_file)
        self.reviews_df = pd.read_csv(self.reviews_file)
        
        logger.info(f"Loaded {len(self.agents_df)} agents and {len(self.reviews_df)} reviews")
        
        return self.agents_df, self.reviews_df
    
    def clean_agent_data(self) -> pd.DataFrame:
        """Clean and process agent data."""
        logger.info("Cleaning agent data...")
        
        df = self.agents_df.copy()
        
        # Parse dates
        df['last_sale_date'] = pd.to_datetime(df['last_sale_date'], errors='coerce')
        
        # Calculate experience years if missing
        current_year = self.current_date.year
        df['experience_years'] = df['experience_years'].fillna(
            current_year - df['first_year_active']
        )
        
        # Clean agent type
        df['agent_type'] = df['agent_type'].fillna('buyer, seller')
        df['is_buyer_agent'] = df['agent_type'].str.contains('buyer', case=False, na=False)
        df['is_seller_agent'] = df['agent_type'].str.contains('seller', case=False, na=False)
        
        # Clean languages
        df['languages'] = df['languages'].fillna('English')
        df['primary_language'] = df['languages'].str.split(',').str[0].str.strip()
        
        # Price ranges for availability matching
        df['price_range_min'] = np.minimum(
            df['active_listings_min_price'].fillna(0),
            df['recently_sold_min_price'].fillna(0)
        )
        df['price_range_max'] = np.maximum(
            df['active_listings_max_price'].fillna(0), 
            df['recently_sold_max_price'].fillna(0)
        )
        
        # Fill missing specializations
        df['specializations'] = df['specializations'].fillna('General Real Estate')
        
        return df
    
    def clean_review_data(self) -> pd.DataFrame:
        """Clean and process review data."""
        logger.info("Cleaning review data...")
        
        df = self.reviews_df.copy()
        
        # Parse dates
        df['review_created_date'] = pd.to_datetime(df['review_created_date'], errors='coerce')
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        
        # Calculate review age in days
        df['review_age_days'] = (self.current_date - df['review_created_date']).dt.days
        
        # Calculate recency weight (exponential decay with 1-year half-life)
        df['recency_weight'] = np.exp(-df['review_age_days'] / 365.0)
        
        # Clean reviewer role
        df['reviewer_role'] = df['reviewer_role'].fillna('BUYER')
        df['is_buyer_review'] = df['reviewer_role'] == 'BUYER'
        df['is_seller_review'] = df['reviewer_role'] == 'SELLER'
        
        # Sentiment proxy (4+ stars = positive)
        df['is_positive'] = df['review_rating'] >= 4.0
        
        # Clean review comments
        df['has_comment'] = df['review_comment'].notna()
        df['review_comment'] = df['review_comment'].fillna('')
        
        # Remove reviews without ratings
        df = df.dropna(subset=['review_rating'])
        
        logger.info(f"Cleaned reviews: {len(df)} remaining")
        
        return df
    
    def calculate_agent_aggregates(self, agents_df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate statistics for each agent from their reviews."""
        logger.info("Calculating agent aggregates from reviews...")
        
        # Group reviews by agent
        agent_stats = []
        
        for agent_id in agents_df['advertiser_id']:
            agent_reviews = reviews_df[reviews_df['advertiser_id'] == agent_id]
            
            if len(agent_reviews) == 0:
                # No reviews - use defaults
                stats = {
                    'advertiser_id': agent_id,
                    'review_count_actual': 0,
                    'avg_rating': 0.0,
                    'avg_responsiveness': np.nan,
                    'avg_negotiation': np.nan, 
                    'avg_professionalism': np.nan,
                    'avg_market_expertise': np.nan,
                    'recency_score': 0.0,
                    'positive_review_ratio': 0.0,
                    'wilson_lower_bound': 0.0,
                    'has_recent_reviews': False,
                    'days_since_last_review': np.inf
                }
            else:
                # Calculate statistics
                stats = {
                    'advertiser_id': agent_id,
                    'review_count_actual': len(agent_reviews),
                    'avg_rating': agent_reviews['review_rating'].mean(),
                    'avg_responsiveness': agent_reviews['sub_responsiveness'].mean(),
                    'avg_negotiation': agent_reviews['sub_negotiation'].mean(),
                    'avg_professionalism': agent_reviews['sub_professionalism'].mean(), 
                    'avg_market_expertise': agent_reviews['sub_market_expertise'].mean(),
                    'recency_score': agent_reviews['recency_weight'].mean(),
                    'positive_review_ratio': agent_reviews['is_positive'].mean(),
                    'wilson_lower_bound': self._wilson_lower_bound(
                        agent_reviews['is_positive'].sum(), 
                        len(agent_reviews)
                    ),
                    'has_recent_reviews': (agent_reviews['review_age_days'] <= 365).any(),
                    'days_since_last_review': agent_reviews['review_age_days'].min()
                }
            
            agent_stats.append(stats)
        
        stats_df = pd.DataFrame(agent_stats)
        
        # Merge with agent data
        result = agents_df.merge(stats_df, on='advertiser_id', how='left')
        
        logger.info(f"Calculated aggregates for {len(result)} agents")
        
        return result
    
    def learn_quality_blend_weights(self, agent_stats_df: pd.DataFrame) -> Dict[str, float]:
        """
        Learn optimal Q_prior blend weights from data using regression.
        
        Your original formula: Q_prior = w1*posterior_mean + w2*wilson_lb + w3*recency
        Instead of hardcoded (0.5, 0.3, 0.2), learn from data.
        
        Args:
            agent_stats_df: DataFrame with calculated agent statistics
            
        Returns:
            Learned blend weights dictionary
        """
        logger.info("Learning Q_prior blend weights from data...")
        
        # Prepare features for learning
        features_df = agent_stats_df[['posterior_mean_rating', 'wilson_lower_bound', 'recency_score']].fillna(0)
        
        # Target: Use overall agent rating as proxy for quality
        # In practice, you could use conversion rates or customer satisfaction
        target = agent_stats_df['agent_rating'].fillna(agent_stats_df['agent_rating'].mean())
        
        # Filter valid cases (agents with some data)
        valid_mask = (features_df.sum(axis=1) > 0) & (target > 0)
        X = features_df[valid_mask]
        y = target[valid_mask]
        
        if len(X) < 10:  # Fallback to default if insufficient data
            logger.warning("Insufficient data for learning blend weights, using defaults")
            return {'posterior_mean': 0.5, 'wilson_bound': 0.3, 'recency': 0.2}
        
        # Fit Ridge regression to learn blend weights
        ridge = Ridge(alpha=1.0, positive=True)  # Positive weights only
        ridge.fit(X, y)
        
        # Extract learned weights
        raw_weights = ridge.coef_
        
        # Normalize to sum to 1
        if raw_weights.sum() > 0:
            normalized_weights = raw_weights / raw_weights.sum()
        else:
            # Fallback to equal weights if all coefficients are zero
            logger.warning("All learned weights are zero, using equal blend weights")
            normalized_weights = np.array([1/3, 1/3, 1/3])
        
        learned_weights = {
            'posterior_mean': float(normalized_weights[0]),
            'wilson_bound': float(normalized_weights[1]), 
            'recency': float(normalized_weights[2])
        }
        
        # Cross-validation score to validate learning
        cv_scores = cross_val_score(ridge, X, y, cv=3, scoring='r2')
        
        logger.info(f"Learned Q_prior weights: {learned_weights}")
        logger.info(f"Cross-validation R² score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return learned_weights
    
    def add_bayesian_rating_shrinkage(self, agent_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Bayesian rating shrinkage as specified in your original proposal.
        
        Posterior mean ~ weighted blend of agent mean and global mean, 
        weight ∝ review count.
        
        Args:
            agent_stats_df: DataFrame with agent statistics
            
        Returns:
            DataFrame with added posterior_mean_rating column
        """
        logger.info("Adding Bayesian rating shrinkage for overall ratings...")
        
        result = agent_stats_df.copy()
        
        # Global prior (overall mean rating across all agents)
        global_mean_rating = result['agent_rating'].fillna(0).mean()
        if global_mean_rating == 0:
            global_mean_rating = 3.5  # Reasonable default
        
        # Shrinkage parameter (higher review count = less shrinkage)
        # Use empirical Bayes approach
        review_counts = result['review_count_actual'].fillna(0)
        
        # Shrinkage weight: more reviews = higher weight on agent's own rating
        # Using tau = mean review count as regularization
        tau = review_counts.mean() if review_counts.mean() > 0 else 10.0
        shrinkage_weights = review_counts / (review_counts + tau)
        
        # Posterior mean calculation
        agent_ratings = result['agent_rating'].fillna(global_mean_rating)
        posterior_means = (
            shrinkage_weights * agent_ratings + 
            (1 - shrinkage_weights) * global_mean_rating
        )
        
        result['posterior_mean_rating'] = posterior_means
        result['shrinkage_weight'] = shrinkage_weights
        
        logger.info(f"Applied Bayesian shrinkage with τ={tau:.1f}, global_mean={global_mean_rating:.2f}")
        
        return result
    
    def calculate_learned_quality_prior(self, agent_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Q_prior using learned blend weights instead of hardcoded values.
        
        Your original proposal: Learn blend coefficients from data, not hardcode them.
        
        Args:
            agent_stats_df: DataFrame with agent statistics including posterior_mean_rating
            
        Returns:
            DataFrame with learned Q_prior column
        """
        # First add Bayesian rating shrinkage
        enhanced_df = self.add_bayesian_rating_shrinkage(agent_stats_df)
        
        # Learn blend weights from data
        blend_weights = self.learn_quality_blend_weights(enhanced_df)
        
        # Calculate Q_prior using learned weights
        q_prior = (
            blend_weights['posterior_mean'] * enhanced_df['posterior_mean_rating'].fillna(0) +
            blend_weights['wilson_bound'] * enhanced_df['wilson_lower_bound'].fillna(0) +
            blend_weights['recency'] * enhanced_df['recency_score'].fillna(0)
        )
        
        enhanced_df['quality_prior'] = q_prior
        enhanced_df['blend_weights'] = str(blend_weights)  # Store for reference
        
        logger.info(f"Calculated learned Q_prior for {len(enhanced_df)} agents")
        
        return enhanced_df
    
    def _wilson_lower_bound(self, positive_count: int, total_count: int, confidence: float = 0.95) -> float:
        """Calculate Wilson score lower bound for confidence interval."""
        if total_count == 0:
            return 0.0
        
        # Z-score for confidence level
        z = 1.96 if confidence == 0.95 else 1.645  # 95% or 90%
        
        p = positive_count / total_count
        n = total_count
        
        denominator = 1 + z**2 / n
        numerator = p + z**2 / (2*n) - z * np.sqrt((p*(1-p) + z**2/(4*n)) / n)
        
        return max(0.0, numerator / denominator)
    
    def apply_empirical_bayes_shrinkage(self, agents_df: pd.DataFrame) -> pd.DataFrame:
        """Apply empirical Bayes shrinkage to sub-scores to prevent overfitting from few reviews."""
        logger.info("Applying empirical Bayes shrinkage to sub-scores...")
        
        df = agents_df.copy()
        
        sub_scores = ['avg_responsiveness', 'avg_negotiation', 'avg_professionalism', 'avg_market_expertise']
        
        for score in sub_scores:
            # Get valid data
            valid_data = df[df[score].notna()]
            
            if len(valid_data) == 0:
                continue
            
            # Calculate global statistics
            global_mean = valid_data[score].mean()
            global_var = valid_data[score].var()
            
            # Estimate prior parameters (method of moments)
            # Assume review count affects uncertainty
            review_counts = valid_data['review_count_actual']
            
            # Shrinkage factor: higher review count = less shrinkage
            # Fix the buggy k calculation - use a reasonable fixed value or compute properly
            if global_var > 0:
                # Use variance-based estimation or fixed reasonable value
                k = max(5.0, min(20.0, 1.0 / global_var * 10))  # Bounded between 5-20
            else:
                k = 10.0  # Default fallback
            
            shrinkage_factors = k / (k + review_counts)
            
            # Apply shrinkage: posterior = (1-shrinkage)*sample + shrinkage*prior
            shrunk_scores = (1 - shrinkage_factors) * valid_data[score] + shrinkage_factors * global_mean
            
            # Update the dataframe
            new_col = f"{score}_shrunk"
            df[new_col] = np.nan
            df.loc[valid_data.index, new_col] = shrunk_scores
            
            # For agents with no reviews, use global mean
            df[new_col] = df[new_col].fillna(global_mean)
        
        return df
    
    def process_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete data processing pipeline."""
        logger.info("Starting complete data processing pipeline...")
        
        # Load data
        self.load_data()
        
        # Clean both datasets
        agents_clean = self.clean_agent_data()
        reviews_clean = self.clean_review_data()
        
        # Calculate aggregates
        agents_with_stats = self.calculate_agent_aggregates(agents_clean, reviews_clean)
        
        # Apply Bayesian shrinkage
        agents_final = self.apply_empirical_bayes_shrinkage(agents_with_stats)
        
        logger.info("Data processing pipeline completed successfully")
        
        return agents_final, reviews_clean
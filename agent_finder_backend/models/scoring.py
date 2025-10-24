import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from utils.stats import aggregate_scores
from config.settings import settings


class AgentScorer:
    """Score and rank agents based on multiple factors."""
    
    def __init__(self):
        """Initialize scorer with default weights."""
        self.base_weights = {
            'buyer_seller_fit': settings.WEIGHT_BUYER_SELLER_FIT,
            'performance': settings.WEIGHT_PERFORMANCE,
            'reviews': settings.WEIGHT_REVIEWS,
            'sub_scores': settings.WEIGHT_SUB_SCORES,
            'proximity': settings.WEIGHT_PROXIMITY,
            'active_listings': settings.WEIGHT_ACTIVE_LISTINGS
        }
    
    def calculate_buyer_seller_score(
        self,
        agent_row: pd.Series,
        user_type: str
    ) -> float:
        """
        Calculate how suitable an agent is for buyer or seller.
        
        Args:
            agent_row: Agent data row
            user_type: 'buyer' or 'seller'
        
        Returns:
            Suitability score (0-1)
        """
        if user_type == 'buyer':
            # Use buyer satisfaction and review count
            satisfaction = agent_row.get('buyer_satisfaction', 0.5)
            review_count = agent_row.get('buyer_review_count', 0)
        else:
            satisfaction = agent_row.get('seller_satisfaction', 0.5)
            review_count = agent_row.get('seller_review_count', 0)
        
        # If agent has both types, check if they claim to be both
        agent_type = str(agent_row.get('agent_type', '')).lower()
        is_both = 'buyer' in agent_type and 'seller' in agent_type
        
        # Base score from satisfaction
        score = satisfaction
        
        # Bonus if agent specializes in this type
        if review_count > 5:
            score *= 1.1  # 10% bonus for experience
        
        # Small bonus if they handle both
        if is_both:
            score *= 1.05
        
        return min(score, 1.0)
    
    def calculate_performance_score(self, agent_row: pd.Series) -> float:
        """
        Calculate overall performance score.
        
        Combines:
        - Recency of sales
        - Volume of recent sales
        - Experience
        
        Args:
            agent_row: Agent data row
        
        Returns:
            Performance score (0-1)
        """
        # Get pre-calculated performance score (recency + volume)
        recency_volume_score = agent_row.get('performance_score', 0)
        
        # Get experience score
        experience_score = agent_row.get('experience_score', 0)
        
        # Combine with weights
        performance = (
            settings.PERFORMANCE_WEIGHT_RECENCY * recency_volume_score +
            settings.PERFORMANCE_WEIGHT_VOLUME * recency_volume_score +
            settings.PERFORMANCE_WEIGHT_EXPERIENCE * experience_score
        )
        
        # Normalize
        performance = performance / (
            settings.PERFORMANCE_WEIGHT_RECENCY +
            settings.PERFORMANCE_WEIGHT_VOLUME +
            settings.PERFORMANCE_WEIGHT_EXPERIENCE
        )
        
        return performance
    
    def calculate_review_score(self, agent_row: pd.Series) -> float:
        """
        Calculate review-based score.
        
        Uses Wilson score and shrunk rating.
        
        Args:
            agent_row: Agent data row
        
        Returns:
            Review score (0-1)
        """
        # Wilson score accounts for positive/negative ratio with confidence
        wilson = agent_row.get('wilson_score', 0)
        
        # Shrunk rating accounts for rating with confidence
        shrunk_rating = agent_row.get('shrunk_rating', 4.0)
        normalized_rating = (shrunk_rating - 1) / 4  # Convert 1-5 to 0-1
        
        # Combine (average)
        review_score = (wilson + normalized_rating) / 2
        
        return review_score
    
    def calculate_sub_scores_weighted(
        self,
        agent_row: pd.Series,
        user_preferences: Dict[str, float]
    ) -> float:
        """
        Calculate weighted sub-scores based on user preferences.
        
        Args:
            agent_row: Agent data row
            user_preferences: User weights for each sub-score
        
        Returns:
            Weighted sub-score (0-1)
        """
        sub_score_mapping = {
            'responsiveness': 'avg_sub_responsiveness',
            'negotiation': 'avg_sub_negotiation',
            'professionalism': 'avg_sub_professionalism',
            'market_expertise': 'avg_sub_market_expertise'
        }
        
        scores = []
        weights = []
        
        for pref_name, weight in user_preferences.items():
            col_name = sub_score_mapping.get(pref_name)
            if col_name and pd.notna(agent_row.get(col_name)):
                # Convert 1-5 rating to 0-1
                score = (agent_row[col_name] - 1) / 4
                scores.append(score)
                weights.append(weight)
        
        if not scores:
            # No sub-scores available, return neutral
            return 0.5
        
        # Weighted average
        weights = np.array(weights)
        scores = np.array(scores)
        
        return np.average(scores, weights=weights)
    
    def calculate_skills_score(
        self,
        agent_row: pd.Series,
        user_skill_preferences: Dict[str, float]
    ) -> float:
        """
        Calculate skills score based on user preferences.
        
        Args:
            agent_row: Agent data row
            user_skill_preferences: User weights for each skill
        
        Returns:
            Weighted skills score (0-1)
        """
        scores = []
        weights = []
        
        for skill_name, weight in user_skill_preferences.items():
            col_name = f'skill_{skill_name}'
            if col_name in agent_row.index and pd.notna(agent_row[col_name]):
                scores.append(agent_row[col_name])
                weights.append(weight)
        
        if not scores:
            return 0.5  # Neutral if no skills available
        
        weights = np.array(weights)
        scores = np.array(scores)
        
        return np.average(scores, weights=weights)
    
    def calculate_negative_quality_penalty(
        self, 
        agent_row: pd.Series
    ) -> float:
        """
        Calculate penalty from negative qualities detected in reviews.
        
        Different negative qualities have different severity levels.
        Multiple negatives compound the penalty.
        
        Args:
            agent_row: Agent data row with negative_* columns
        
        Returns:
            Penalty multiplier (0-1, where 1 = no penalty)
        """
        # Severity weights for each negative quality
        severity_penalties = {
            'negative_dishonest': 0.60,      # 40% penalty (most severe)
            'negative_unprofessional': 0.75,  # 25% penalty
            'negative_unresponsive': 0.80,    # 20% penalty
            'negative_pushy': 0.85,           # 15% penalty
            'negative_inexperienced': 0.90    # 10% penalty (least severe)
        }
        
        total_penalty = 1.0
        
        for negative_col, base_penalty in severity_penalties.items():
            if negative_col in agent_row.index:
                score = agent_row[negative_col]
                
                # Only apply penalty if confidence is above threshold
                if pd.notna(score) and score > 0.5:
                    # Scale penalty by confidence
                    # score=0.5 → minimal penalty
                    # score=1.0 → full penalty
                    confidence_factor = (score - 0.5) / 0.5
                    penalty_factor = 1 - (1 - base_penalty) * confidence_factor
                    total_penalty *= penalty_factor
        
        return max(total_penalty, 0.5)  # Cap minimum at 50% of original score
    
    def calculate_active_listings_score(
        self,
        agent_row: pd.Series,
        is_urgent: bool
    ) -> float:
        """
        Calculate score based on active listings (only for urgent users).
        
        Args:
            agent_row: Agent data row
            is_urgent: Whether user has urgent need
        
        Returns:
            Active listings score (0-1)
        """
        if not is_urgent:
            return 0.5  # Neutral if not urgent
        
        active_count = agent_row.get('active_listings_count', 0)
        
        # Normalize (log scale)
        score = np.log1p(active_count) / np.log1p(50)  # Max expected ~50
        
        return min(score, 1.0)
    
    def score_agents(
        self,
        agents_df: pd.DataFrame,
        user_type: str,
        user_preferences: Dict[str, float],
        user_skill_preferences: Dict[str, float],
        is_urgent: bool,
        proximity_scores: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate comprehensive scores for all agents.
        
        Args:
            agents_df: DataFrame with agent data
            user_type: 'buyer' or 'seller'
            user_preferences: User weights for sub-scores
            user_skill_preferences: User weights for skills
            is_urgent: Whether user has urgent need
            proximity_scores: Pre-calculated proximity scores (optional)
        
        Returns:
            DataFrame with added score columns
        """
        # Calculate individual score components
        agents_df['buyer_seller_score'] = agents_df.apply(
            lambda row: self.calculate_buyer_seller_score(row, user_type),
            axis=1
        )
        
        agents_df['performance_component'] = agents_df.apply(
            self.calculate_performance_score,
            axis=1
        )
        
        agents_df['review_component'] = agents_df.apply(
            self.calculate_review_score,
            axis=1
        )
        
        agents_df['sub_scores_component'] = agents_df.apply(
            lambda row: self.calculate_sub_scores_weighted(row, user_preferences),
            axis=1
        )
        
        agents_df['skills_component'] = agents_df.apply(
            lambda row: self.calculate_skills_score(row, user_skill_preferences),
            axis=1
        )
        
        agents_df['negative_penalty'] = agents_df.apply(
            self.calculate_negative_penalty,
            axis=1
        )
        
        agents_df['active_listings_score'] = agents_df.apply(
            lambda row: self.calculate_active_listings_score(row, is_urgent),
            axis=1
        )
        
        # Add proximity scores if provided
        if proximity_scores is not None:
            agents_df['proximity_score'] = proximity_scores
        else:
            agents_df['proximity_score'] = 0.5  # Neutral
        
        # Adjust weights if urgent
        weights = self.base_weights.copy()
        if is_urgent:
            # Increase weight for active listings
            weights['active_listings'] = 0.15
            # Redistribute others
            weights['buyer_seller_fit'] = 0.18
            weights['performance'] = 0.22
            weights['reviews'] = 0.18
            weights['sub_scores'] = 0.12
            weights['proximity'] = 0.10
        
        # Combine components with skills mixed into reviews
        combined_review = (
            0.6 * agents_df['review_component'] +
            0.4 * agents_df['skills_component']
        )
        
        # Calculate final matching score (before penalty)
        score_components = np.column_stack([
            agents_df['buyer_seller_score'],
            agents_df['performance_component'],
            combined_review,
            agents_df['sub_scores_component'],
            agents_df['proximity_score'],
            agents_df['active_listings_score']
        ])
        
        weight_array = np.array([
            weights['buyer_seller_fit'],
            weights['performance'],
            weights['reviews'],
            weights['sub_scores'],
            weights['proximity'],
            weights['active_listings']
        ])
        
        # Weighted average
        agents_df['matching_score_raw'] = np.average(
            score_components,
            weights=weight_array,
            axis=1
        )
        
        # Apply negative penalty
        agents_df['matching_score'] = (
            agents_df['matching_score_raw'] * agents_df['negative_penalty']
        )
        
        # Convert to 0-100 scale for display
        agents_df['matching_score_display'] = (
            agents_df['matching_score'] * 100
        ).round(1)
        
        return agents_df


# Global scorer instance
agent_scorer = AgentScorer()
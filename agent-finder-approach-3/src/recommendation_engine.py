"""
Agent recommendation engine with personalized scoring and ranking.

Implements:
- Personalized utility function computation
- Availability fit scoring based on user requirements
- MMR (Maximal Marginal Relevance) for diversification
- Wilson confidence-based filtering
- Final agent ranking and selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Main recommendation engine for agent ranking and selection."""
    
    def __init__(self, 
                 diversity_lambda: float = 0.3,
                 min_confidence: float = 0.1,
                 max_results: int = 50):
        """
        Initialize recommendation engine.
        
        Args:
            diversity_lambda: Balance between relevance and diversity in MMR (0-1)
            min_confidence: Minimum confidence threshold for agent inclusion
            max_results: Maximum number of agents to return
        """
        self.diversity_lambda = diversity_lambda
        self.min_confidence = min_confidence
        self.max_results = max_results
        
        # Learned parameters (set during training)
        self.gamma1 = 0.05  # Weight for Q_prior (further reduced to prioritize personalized skills)
        self.gamma2 = 0.02  # Weight for availability fit (further reduced)

    def apply_hard_filters(self, agents_df: pd.DataFrame, user_filters: Dict) -> pd.DataFrame:
        """
        Apply hard filters first - agents must meet these criteria exactly.
        
        Args:
            agents_df: Agent data
            user_filters: User filter criteria
            
        Returns:
            Filtered agent dataframe
        """
        logger.info(f"Applying hard filters to {len(agents_df)} agents...")
        
        filtered_df = agents_df.copy()
        
        # 1. State filtering (critical - must match exactly)
        if 'state' in user_filters and user_filters['state']:
            state_matches = filtered_df['state'] == user_filters['state']
            filtered_df = filtered_df[state_matches]
            logger.info(f"After state filter ({user_filters['state']}): {len(filtered_df)} agents")
        
        # 2. Language filtering (critical for non-English)
        if 'language' in user_filters and user_filters['language'] and user_filters['language'].lower() != 'english':
            # For non-English languages, require exact match
            lang_matches = filtered_df['languages'].fillna('English').str.contains(
                user_filters['language'], case=False, na=False
            )
            filtered_df = filtered_df[lang_matches]
            logger.info(f"After language filter ({user_filters['language']}): {len(filtered_df)} agents")
        
        # 3. Transaction type filtering
        if 'transaction_type' in user_filters and user_filters['transaction_type']:
            if user_filters['transaction_type'].lower() == 'buying':
                type_matches = filtered_df['is_buyer_agent'].fillna(False)
            elif user_filters['transaction_type'].lower() == 'selling':
                type_matches = filtered_df['is_seller_agent'].fillna(False)
            else:
                type_matches = pd.Series(True, index=filtered_df.index)
            
            filtered_df = filtered_df[type_matches]
            logger.info(f"After transaction type filter ({user_filters['transaction_type']}): {len(filtered_df)} agents")
        
        # 4. Rating filtering (be more lenient for non-English speakers)
        if 'min_rating' in user_filters and user_filters['min_rating']:
            min_rating = user_filters['min_rating']
            # If requesting non-English language, prioritize language match over rating
            if 'language' in user_filters and user_filters['language'].lower() != 'english':
                # For rare languages, accept agents with any rating >= 0
                min_rating = 0.0
                logger.info(f"Prioritizing language match for {user_filters['language']} speakers - no rating requirement")
            
            rating_matches = filtered_df['agent_rating'].fillna(0) >= min_rating
            filtered_df = filtered_df[rating_matches]
            logger.info(f"After rating filter (>={min_rating}): {len(filtered_df)} agents")
        
        # 5. Review count filtering (be more lenient for non-English speakers)
        if 'min_reviews' in user_filters and user_filters['min_reviews']:
            min_reviews = user_filters['min_reviews']
            # If requesting non-English language, prioritize language match over review count
            if 'language' in user_filters and user_filters['language'].lower() != 'english':
                min_reviews = 0  # Accept any review count for rare languages
                logger.info(f"Prioritizing language match for {user_filters['language']} speakers - no review requirement")
            
            review_matches = filtered_df['review_count_actual'].fillna(0) >= min_reviews
            filtered_df = filtered_df[review_matches]
            logger.info(f"After review count filter (>={min_reviews}): {len(filtered_df)} agents")
        
        # 6. Active status filtering (more lenient for non-English speakers)
        if user_filters.get('active_only', True):
            # For non-English speakers, skip activity filtering entirely to prioritize language match
            if 'language' in user_filters and user_filters['language'].lower() != 'english':
                logger.info(f"Skipping activity filter for {user_filters['language']} speakers to prioritize language match")
            else:
                activity_threshold = 0.1
                recent_activity = filtered_df['recency_score'].fillna(0) > activity_threshold
                filtered_df = filtered_df[recent_activity]
            logger.info(f"After active filter: {len(filtered_df)} agents")
        
        # 7. Recent activity filtering (more lenient for non-English speakers)
        if user_filters.get('require_recent_activity', False):
            # For non-English speakers, skip recent activity requirement to prioritize language match
            if 'language' in user_filters and user_filters['language'].lower() != 'english':
                logger.info(f"Skipping recent activity requirement for {user_filters['language']} speakers to prioritize language match")
            else:
                recent_threshold = 0.3
                recent_matches = filtered_df['recency_score'].fillna(0) > recent_threshold
                filtered_df = filtered_df[recent_matches]
            logger.info(f"After recent activity filter: {len(filtered_df)} agents")
        
        logger.info(f"Final filtered pool: {len(filtered_df)} agents")
        
        if len(filtered_df) == 0:
            logger.warning("No agents match the specified filters!")
        
        return filtered_df
    
    def calculate_availability_fit(self,
                                 agents_df: pd.DataFrame,
                                 user_filters: Dict[str, Any]) -> pd.Series:
        """
        Calculate how well each agent fits user availability requirements.
        
        Args:
            agents_df: Agent data with features
            user_filters: User requirements (location, price, agent_type, etc.)
            
        Returns:
            Availability fit scores (0-1) for each agent
        """
        logger.info("Calculating availability fit scores...")
        
        fit_scores = pd.Series(1.0, index=agents_df.index)  # Start with perfect fit
        
        # City matching (soft scoring for service area coverage)
        if 'city' in user_filters and user_filters['city']:
            # Check if user city is in agent's service areas
            city_matches = agents_df['marketing_area_cities'].fillna('').str.contains(
                user_filters['city'], case=False, na=False
            ).astype(float)
            # Also check base city proximity (soft scoring)
            base_city_matches = agents_df['agent_base_city'].fillna('').str.contains(
                user_filters['city'], case=False, na=False
            ).astype(float)
            # Combine with higher weight for service area match
            city_fit = 0.8 * city_matches + 0.6 * base_city_matches
            city_fit = np.minimum(city_fit, 1.0)  # Cap at 1.0
            fit_scores *= city_fit
        
        # Price range matching (soft scoring)
        if 'price_min' in user_filters or 'price_max' in user_filters:
            price_fit = self._calculate_price_fit(
                agents_df, 
                user_filters.get('price_min'),
                user_filters.get('price_max')
            )
            fit_scores *= price_fit
        
        # Specialization matching (soft boost)
        if 'specialization' in user_filters and user_filters['specialization']:
            spec_match = agents_df['specializations'].fillna('').str.contains(
                user_filters['specialization'], case=False, na=False
            ).astype(float)
            # Boost matching agents, don't penalize non-matching
            fit_scores = fit_scores + spec_match * 0.2
            fit_scores = np.minimum(fit_scores, 1.0)
        
        return fit_scores
    
    def _calculate_price_fit(self, 
                           agents_df: pd.DataFrame, 
                           price_min: Optional[float], 
                           price_max: Optional[float]) -> pd.Series:
        """Calculate price range compatibility."""
        
        fit_scores = pd.Series(1.0, index=agents_df.index)
        
        if price_min is None and price_max is None:
            return fit_scores
        
        # Use agent's price range from listings and sales
        agent_min_prices = agents_df['price_range_min'].fillna(0)
        agent_max_prices = agents_df['price_range_max'].fillna(float('inf'))
        
        if price_min is not None:
            # Penalize if agent's max is much lower than user's min
            low_penalty = np.where(
                agent_max_prices < price_min * 0.8,  # 20% tolerance
                0.5,  # Half penalty
                1.0
            )
            fit_scores *= low_penalty
        
        if price_max is not None:
            # Penalize if agent's min is much higher than user's max  
            high_penalty = np.where(
                agent_min_prices > price_max * 1.2,  # 20% tolerance
                0.5,  # Half penalty
                1.0
            )
            fit_scores *= high_penalty
        
        return fit_scores
    
    def calculate_personalized_utility(self,
                                     agents_df: pd.DataFrame,
                                     fused_weights: np.ndarray,
                                     feature_names: List[str],
                                     availability_fit: pd.Series) -> pd.Series:
        """
        Calculate personalized utility scores for agents.
        
        U = β' · [sub-scores, theme strengths] + γ1 * Q_prior + γ2 * availability_fit
        
        Args:
            agents_df: Agent data with skill vectors and Q_prior
            fused_weights: Personalized feature weights
            feature_names: Names of features corresponding to weights
            availability_fit: Availability fit scores
            
        Returns:
            Utility scores for each agent
        """
        logger.info("Calculating personalized utility scores...")
        
        # Extract feature matrix
        feature_matrix = []
        for _, agent in agents_df.iterrows():
            agent_features = []
            for feature_name in feature_names:
                agent_features.append(agent.get(feature_name, 0.0))
            feature_matrix.append(agent_features)
        
        feature_matrix = np.array(feature_matrix)
        
        # Calculate main utility component: β' · features + PERSONALIZED component
        main_utility = np.dot(feature_matrix, fused_weights)
        
        # Debug logging for utility calculation
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        logger.info(f"Fused weights shape: {fused_weights.shape}")
        logger.info(f"Main utility stats: min={main_utility.min():.3f}, max={main_utility.max():.3f}, mean={main_utility.mean():.3f}")
        logger.info(f"Base weights magnitude: {np.linalg.norm(fused_weights):.3f}")
        
        # DIRECT PERSONALIZATION: Add personalized skill component
        # Extract user preferences for direct application
        user_prefs = getattr(self, '_current_user_preferences', {})
        if user_prefs:
            # Direct skill-based personalization
            personalized_component = (
                user_prefs.get('responsiveness', 0.5) * agents_df.get('skill_responsiveness', 0).fillna(0) +
                user_prefs.get('negotiation', 0.5) * agents_df.get('skill_negotiation', 0).fillna(0) +
                user_prefs.get('professionalism', 0.5) * agents_df.get('skill_professionalism', 0).fillna(0) +
                user_prefs.get('market_expertise', 0.5) * agents_df.get('skill_market_expertise', 0).fillna(0)
            ) * 5.0  # Scale to have significant impact
        else:
            personalized_component = 0
        
        # Add penalty for agents with very low skill scores
        skill_columns = ['skill_responsiveness', 'skill_negotiation', 'skill_professionalism', 'skill_market_expertise']
        skill_scores = agents_df[skill_columns].fillna(0.0)
        
        # If all skills are near zero, apply penalty
        skill_sum = skill_scores.sum(axis=1)
        zero_skill_penalty = np.where(skill_sum < 0.5, -2.0, 0.0)  # Strong penalty for zero-skill agents
        
        # Add Q_prior component
        q_prior = agents_df['q_prior'].fillna(0.0)
        
        # Add availability fit component
        availability_component = availability_fit.reindex(agents_df.index, fill_value=0.0)
        
        # Combine components
        utility_scores = main_utility + personalized_component + self.gamma1 * q_prior + self.gamma2 * availability_component + zero_skill_penalty
        
        return pd.Series(utility_scores, index=agents_df.index)
    
    def apply_confidence_filtering(self, 
                                 agents_df: pd.DataFrame,
                                 utility_scores: pd.Series,
                                 user_filters: Dict = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Filter agents based on confidence thresholds.
        
        Args:
            agents_df: Agent data with confidence metrics
            utility_scores: Utility scores for agents
            user_filters: User preferences and filters
            
        Returns:
            Filtered agents and utility scores
        """
        # For language-specific requests (non-English), SKIP confidence filtering
        # Language matching is the absolute priority
        if (user_filters and 
            'language' in user_filters and 
            user_filters['language'] and 
            user_filters['language'].lower() != 'english'):
            logger.info(f"🌍 LANGUAGE-PRIORITY: Skipping confidence filtering for {user_filters['language']} speakers")
            return agents_df.copy(), utility_scores
        
        # Use Wilson lower bound as confidence measure
        confidence_scores = agents_df['wilson_lower_bound'].fillna(0.0)
        
        # Filter by minimum confidence
        confident_mask = confidence_scores >= self.min_confidence
        
        # Also require some minimum data (at least 1 review or substantial profile)
        data_mask = (
            (agents_df['review_count_actual'] >= 1) |
            (agents_df.get('data_richness', 0) >= 0.3)
        )
        
        # Combine filters
        valid_mask = confident_mask & data_mask
        
        logger.info(f"Confidence filtering: {valid_mask.sum()}/{len(agents_df)} agents passed")
        
        filtered_agents = agents_df[valid_mask].copy()
        filtered_utility = utility_scores[valid_mask]
        
        return filtered_agents, filtered_utility
    
    def apply_mmr_diversification(self,
                                agents_df: pd.DataFrame,
                                utility_scores: pd.Series,
                                theme_columns: List[str],
                                top_k: int = 20) -> List[int]:
        """
        Apply Maximal Marginal Relevance (MMR) for diverse recommendations.
        
        Args:
            agents_df: Agent data with theme vectors
            utility_scores: Utility scores for ranking
            theme_columns: Names of theme feature columns
            top_k: Number of diverse agents to select
            
        Returns:
            List of agent advertiser_ids in diversified order
        """
        logger.info("Applying MMR diversification...")
        
        # Get theme vectors for similarity calculation
        if not theme_columns:
            # No themes available, fall back to utility ranking
            sorted_agents = utility_scores.sort_values(ascending=False).head(top_k)
            return sorted_agents.index.tolist()
        
        theme_matrix = agents_df[theme_columns].fillna(0.0).values
        
        # Reset index to ensure we work with positional indices consistently
        agents_reset = agents_df.reset_index(drop=True)
        utility_reset = utility_scores.reindex(agents_df.index).reset_index(drop=True)
        
        # MMR algorithm using positional indices
        selected_advertiser_ids = []
        remaining_positions = set(range(len(agents_reset)))
        
        # Start with highest utility agent
        if len(remaining_positions) > 0:
            best_pos = utility_reset.argmax()
            selected_advertiser_ids.append(agents_reset.iloc[best_pos]['advertiser_id'])
            remaining_positions.remove(best_pos)
        
        # Iteratively select diverse agents
        while len(selected_advertiser_ids) < top_k and remaining_positions:
            best_mmr_score = -float('inf')
            best_pos = None
            
            for pos in remaining_positions:
                # Relevance score (normalized utility)
                relevance = utility_reset.iloc[pos]
                
                # Diversity score (minimum similarity to selected agents)
                if len(selected_advertiser_ids) > 0:
                    # Get positions of already selected agents
                    selected_positions = []
                    for sel_id in selected_advertiser_ids:
                        sel_pos = agents_reset[agents_reset['advertiser_id'] == sel_id].index[0]
                        selected_positions.append(sel_pos)
                    
                    selected_theme_vectors = theme_matrix[selected_positions]
                    current_theme_vector = theme_matrix[pos].reshape(1, -1)
                    
                    # Calculate similarities
                    similarities = cosine_similarity(current_theme_vector, selected_theme_vectors)[0]
                    max_similarity = np.max(similarities) if len(similarities) > 0 else 0
                    diversity = 1 - max_similarity
                else:
                    diversity = 1.0
                
                # MMR score: λ * relevance + (1-λ) * diversity
                mmr_score = (self.diversity_lambda * relevance + 
                           (1 - self.diversity_lambda) * diversity)
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_pos = pos
            
            if best_pos is not None:
                selected_advertiser_ids.append(agents_reset.iloc[best_pos]['advertiser_id'])
                remaining_positions.remove(best_pos)
        
        logger.info(f"Selected {len(selected_advertiser_ids)} diverse agents")
        
        return selected_advertiser_ids
    
    def rank_and_recommend(self,
                          agents_df: pd.DataFrame,
                          fused_weights: np.ndarray,
                          feature_names: List[str],
                          theme_columns: List[str],
                          user_filters: Dict[str, Any],
                          top_k: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete recommendation pipeline: score, filter, diversify, and rank agents.
        
        Args:
            agents_df: Agent data with all features
            fused_weights: Personalized feature weights
            feature_names: Names of features
            theme_columns: Theme feature column names
            user_filters: User requirements and filters
            top_k: Number of recommendations to return
            
        Returns:
            Ranked agent recommendations and scoring metadata
        """
        if top_k is None:
            top_k = min(self.max_results, len(agents_df))
        
        logger.info(f"Starting recommendation pipeline for {len(agents_df)} agents...")
        
        # Step 0: Apply hard filters first (NEW - prioritize language, state, etc.)
        hard_filtered_agents = self.apply_hard_filters(agents_df, user_filters)
        
        if len(hard_filtered_agents) == 0:
            logger.warning("No agents match the hard filter criteria")
            return pd.DataFrame(), {}
        
        # Step 1: Calculate availability fit on filtered pool
        availability_fit = self.calculate_availability_fit(hard_filtered_agents, user_filters)
        
        # Step 2: Calculate personalized utility on filtered pool
        utility_scores = self.calculate_personalized_utility(
            hard_filtered_agents, fused_weights, feature_names, availability_fit
        )
        
        # Step 3: Apply confidence filtering on hard filtered pool
        filtered_agents, filtered_utility = self.apply_confidence_filtering(
            hard_filtered_agents, utility_scores, user_filters
        )
        
        if len(filtered_agents) == 0:
            logger.warning("No agents passed confidence filtering")
            return pd.DataFrame(), {}
        
        # Step 4: Apply hard constraint filtering
        constraint_filtered_agents, constraint_filtered_utility = self._apply_hard_constraints(
            filtered_agents, filtered_utility, user_filters
        )
        
        if len(constraint_filtered_agents) == 0:
            logger.warning("No agents passed constraint filtering")
            return pd.DataFrame(), {}
        
        # Step 5: Apply MMR diversification
        diversified_indices = self.apply_mmr_diversification(
            constraint_filtered_agents,
            constraint_filtered_utility,
            theme_columns,
            top_k
        )
        
        # Step 6: Create final recommendations
        recommended_agents = constraint_filtered_agents.loc[
            constraint_filtered_agents['advertiser_id'].isin(diversified_indices)
        ].copy()
        
        # Preserve the MMR order by reordering based on diversified_indices
        recommended_agents = recommended_agents.set_index('advertiser_id').reindex(diversified_indices).reset_index()
        
        # Add scoring metadata
        logger.info(f"Recommended agent IDs: {recommended_agents['advertiser_id'].tolist()}")
        logger.info(f"Constraint filtered utility index: {constraint_filtered_utility.index.tolist()}")
        logger.info(f"Constraint filtered utility values: {constraint_filtered_utility.values}")
        
        # Fix: Map utility scores correctly by using the original agent dataframe index
        # constraint_filtered_utility has pandas index from the original agents_df
        # We need to map advertiser_id -> utility_score correctly
        
        recommended_utility_scores = []
        recommended_availability_scores = []
        
        for agent_id in recommended_agents['advertiser_id']:
            # Find the original index for this agent_id in constraint_filtered_agents
            original_indices = constraint_filtered_agents[constraint_filtered_agents['advertiser_id'] == agent_id].index
            
            if len(original_indices) > 0:
                original_idx = original_indices[0]
                utility_score = constraint_filtered_utility.loc[original_idx]
                availability_score = availability_fit.loc[original_idx] if original_idx in availability_fit.index else 0.0
            else:
                logger.warning(f"Could not find original index for agent {agent_id}")
                utility_score = 0.0
                availability_score = 0.0
            
            recommended_utility_scores.append(utility_score)
            recommended_availability_scores.append(availability_score)
        
        logger.info(f"Final utility scores: {recommended_utility_scores}")
        
        recommended_agents['utility_score'] = recommended_utility_scores
        recommended_agents['availability_fit'] = recommended_availability_scores
        recommended_agents['rank'] = range(1, len(recommended_agents) + 1)
        
        # Metadata for explanations
        metadata = {
            'total_agents': len(agents_df),
            'after_confidence_filter': len(filtered_agents),
            'after_constraint_filter': len(constraint_filtered_agents),
            'final_recommendations': len(recommended_agents),
            'diversity_lambda': self.diversity_lambda,
            'user_filters': user_filters,
            'feature_weights': dict(zip(feature_names, fused_weights))
        }
        
        logger.info(f"Recommendation pipeline completed: {len(recommended_agents)} recommendations")
        
        return recommended_agents, metadata
    
    def _apply_hard_constraints(self,
                              agents_df: pd.DataFrame,
                              utility_scores: pd.Series,
                              user_filters: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply hard constraint filtering that must be satisfied."""
        
        # For language-specific requests (non-English), SKIP constraint filtering
        # Language matching is the absolute priority
        if ('language' in user_filters and 
            user_filters['language'] and 
            user_filters['language'].lower() != 'english'):
            logger.info(f"🌍 LANGUAGE-PRIORITY: Skipping constraint filtering for {user_filters['language']} speakers")
            return agents_df.copy(), utility_scores
        
        constraint_mask = pd.Series(True, index=agents_df.index)
        
        # Required minimum rating
        if 'min_rating' in user_filters and user_filters['min_rating']:
            rating_mask = agents_df['agent_rating'] >= user_filters['min_rating']
            constraint_mask &= rating_mask
        
        # Required minimum review count
        if 'min_reviews' in user_filters and user_filters['min_reviews']:
            review_mask = agents_df['review_count_actual'] >= user_filters['min_reviews']
            constraint_mask &= review_mask
        
        # Required recent activity
        if user_filters.get('require_recent_activity', False):
            recent_mask = agents_df.get('has_recent_reviews', True)
            constraint_mask &= recent_mask
        
        # Active agent requirement
        if user_filters.get('active_only', True):
            # Consider agent active if they have listings or recent sales
            # Handle both possible column names for robustness
            if 'days_since_last_sale' in agents_df.columns:
                recent_sales_mask = agents_df['days_since_last_sale'] <= 180
            elif 'days_since_last_review' in agents_df.columns:
                recent_sales_mask = agents_df['days_since_last_review'] <= 180
            else:
                recent_sales_mask = pd.Series(True, index=agents_df.index)
            
            active_mask = (
                (agents_df['active_listings_count'] > 0) |
                recent_sales_mask
            )
            constraint_mask &= active_mask
        
        filtered_agents = agents_df[constraint_mask].copy()
        filtered_utility = utility_scores[constraint_mask]
        
        logger.info(f"Hard constraint filtering: {len(filtered_agents)}/{len(agents_df)} agents passed")
        
        return filtered_agents, filtered_utility
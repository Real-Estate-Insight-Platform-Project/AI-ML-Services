"""
Baseline Agent Recommender using interpretable scoring.

This module implements the baseline scoring formula:
score(agent, query) =
  0.30 * geo_overlap
+ 0.25 * price_band_match
+ 0.15 * property_type_match
+ 0.15 * normalized_recency
+ 0.10 * rating_score
+ 0.03 * log1p(numReviews)
+ 0.02 * partner_premier_boost
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineAgentScorer:
    """Baseline agent scoring using interpretable weighted features."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the baseline scorer with feature weights.
        
        Args:
            weights: Dictionary of feature weights. If None, uses default weights.
        """
        self.weights = weights or {
            'geo_overlap': 0.30,
            'price_band_match': 0.25,
            'property_type_match': 0.15,
            'normalized_recency': 0.15,
            'rating_score': 0.10,
            'log_reviews_normalized': 0.03,
            'partner_premier_boost': 0.02
        }
        
        # Validate weights sum to ~1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_weight:.3f}, not 1.0")
    
    def calculate_baseline_score(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Calculate baseline scores for all agents.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            pandas.Series: Baseline scores for each agent
        """
        scores = pd.Series(0.0, index=features_df.index)
        
        for feature, weight in self.weights.items():
            if feature in features_df.columns:
                # Ensure values are between 0 and 1
                feature_values = features_df[feature].fillna(0)
                feature_values = np.clip(feature_values, 0, 1)
                scores += weight * feature_values
            else:
                logger.warning(f"Feature '{feature}' not found in DataFrame")
        
        return scores
    
    def get_score_breakdown(self, features_df: pd.DataFrame, agent_indices: List[int] = None) -> pd.DataFrame:
        """
        Get detailed score breakdown for specific agents.
        
        Args:
            features_df: DataFrame with extracted features
            agent_indices: List of agent indices to analyze. If None, analyzes all.
            
        Returns:
            pandas.DataFrame: Detailed score breakdown
        """
        if agent_indices is None:
            agent_indices = features_df.index.tolist()
        
        breakdown_data = []
        
        for idx in agent_indices:
            if idx not in features_df.index:
                continue
                
            agent = features_df.loc[idx]
            breakdown = {
                'agentId': agent.get('agentId', idx),
                'name': agent.get('name', 'Unknown'),
                'total_score': 0.0
            }
            
            for feature, weight in self.weights.items():
                if feature in features_df.columns:
                    feature_value = agent.get(feature, 0)
                    feature_value = np.clip(feature_value, 0, 1)
                    weighted_score = weight * feature_value
                    breakdown[f'{feature}_value'] = feature_value
                    breakdown[f'{feature}_weighted'] = weighted_score
                    breakdown['total_score'] += weighted_score
            
            breakdown_data.append(breakdown)
        
        return pd.DataFrame(breakdown_data)
    
    def rank_agents(self, features_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """
        Rank agents using baseline scoring.
        
        Args:
            features_df: DataFrame with extracted features
            top_k: Number of top agents to return
            
        Returns:
            pandas.DataFrame: Top-k ranked agents with scores
        """
        # Calculate baseline scores
        scores = self.calculate_baseline_score(features_df)
        
        # Add scores to DataFrame
        result_df = features_df.copy()
        result_df['baseline_score'] = scores
        
        # Sort by score (descending) and return top-k
        ranked_df = result_df.sort_values('baseline_score', ascending=False).head(top_k)
        
        return ranked_df
    
    def explain_ranking(self, features_df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """
        Provide detailed explanation of the ranking process.
        
        Args:
            features_df: DataFrame with extracted features
            top_k: Number of top agents to explain
            
        Returns:
            dict: Explanation of ranking methodology and results
        """
        # Get top-k agents
        ranked_df = self.rank_agents(features_df, top_k)
        
        # Get score breakdown for top agents
        top_indices = ranked_df.index.tolist()
        breakdown_df = self.get_score_breakdown(features_df, top_indices)
        
        # Calculate feature statistics
        feature_stats = {}
        for feature in self.weights.keys():
            if feature in features_df.columns:
                feature_stats[feature] = {
                    'mean': features_df[feature].mean(),
                    'std': features_df[feature].std(),
                    'min': features_df[feature].min(),
                    'max': features_df[feature].max(),
                    'weight': self.weights[feature]
                }
        
        explanation = {
            'methodology': {
                'description': 'Baseline scoring using weighted combination of interpretable features',
                'formula': 'score = 0.30*geo_overlap + 0.25*price_band_match + 0.15*property_type_match + 0.15*normalized_recency + 0.10*rating_score + 0.03*log_reviews + 0.02*partner_premier_boost',
                'weights': self.weights
            },
            'top_agents': ranked_df[['agentId', 'name', 'baseline_score', 'starRating', 'numReviews', 'pastYearDeals']].to_dict('records'),
            'score_breakdown': breakdown_df.to_dict('records'),
            'feature_statistics': feature_stats,
            'total_agents_evaluated': len(features_df)
        }
        
        return explanation


class BaselineRecommender:
    """Complete baseline recommender system."""
    
    def __init__(self, feature_extractor, scorer: Optional[BaselineAgentScorer] = None):
        """
        Initialize the recommender.
        
        Args:
            feature_extractor: FeatureExtractor instance
            scorer: BaselineAgentScorer instance. If None, uses default.
        """
        self.feature_extractor = feature_extractor
        self.scorer = scorer or BaselineAgentScorer()
    
    def recommend_agents(self, user_query: Dict[str, Any], top_k: int = 10, 
                        explain: bool = False) -> Dict[str, Any]:
        """
        Recommend top-k agents for a user query.
        
        Args:
            user_query: User preferences dictionary
                - regions: List[str] - target regions
                - budget: float - budget amount  
                - property_types: List[str] - desired property types
            top_k: Number of agents to recommend
            explain: Whether to include detailed explanation
            
        Returns:
            dict: Recommendations with optional explanation
        """
        # Extract features for the query
        features_df = self.feature_extractor.extract_features_for_query(user_query)
        
        # Rank agents
        ranked_df = self.scorer.rank_agents(features_df, top_k)
        
        # Prepare recommendations
        recommendations = []
        for _, agent in ranked_df.iterrows():
            rec = {
                'agentId': agent.get('agentId'),
                'name': agent.get('name'),
                'score': agent.get('baseline_score'),
                'starRating': agent.get('starRating'),
                'numReviews': agent.get('numReviews'),
                'pastYearDeals': agent.get('pastYearDeals'),
                'businessMarket': agent.get('businessMarket'),
                'brokerageName': agent.get('brokerageName'),
                'primaryServiceRegions': agent.get('primaryServiceRegions'),
                'propertyTypes': agent.get('propertyTypes'),
                'partner': agent.get('partner'),
                'isPremier': agent.get('isPremier'),
                'email': agent.get('email'),
                'phoneNumber': agent.get('phoneNumber'),
                'profileUrl': agent.get('profileUrl')
            }
            recommendations.append(rec)
        
        result = {
            'query': user_query,
            'recommendations': recommendations,
            'total_agents_evaluated': len(features_df),
            'model_type': 'baseline'
        }
        
        if explain:
            explanation = self.scorer.explain_ranking(features_df, top_k)
            result['explanation'] = explanation
        
        return result
    
    def compare_agents(self, user_query: Dict[str, Any], 
                      agent_ids: List[int]) -> Dict[str, Any]:
        """
        Compare specific agents for a user query.
        
        Args:
            user_query: User preferences dictionary
            agent_ids: List of agent IDs to compare
            
        Returns:
            dict: Comparison results
        """
        # Extract features for the query
        features_df = self.feature_extractor.extract_features_for_query(user_query)
        
        # Filter to requested agents
        agent_mask = features_df['agentId'].isin(agent_ids)
        filtered_df = features_df[agent_mask]
        
        if len(filtered_df) == 0:
            return {'error': 'No agents found with the specified IDs'}
        
        # Get score breakdown
        breakdown_df = self.scorer.get_score_breakdown(filtered_df)
        
        # Calculate scores
        scores = self.scorer.calculate_baseline_score(filtered_df)
        filtered_df['baseline_score'] = scores
        
        # Sort by score
        sorted_df = filtered_df.sort_values('baseline_score', ascending=False)
        
        comparison_data = []
        for _, agent in sorted_df.iterrows():
            agent_breakdown = breakdown_df[
                breakdown_df['agentId'] == agent['agentId']
            ].iloc[0].to_dict() if len(breakdown_df) > 0 else {}
            
            comparison_data.append({
                'agent_info': {
                    'agentId': agent.get('agentId'),
                    'name': agent.get('name'),
                    'starRating': agent.get('starRating'),
                    'numReviews': agent.get('numReviews'),
                    'pastYearDeals': agent.get('pastYearDeals'),
                    'businessMarket': agent.get('businessMarket')
                },
                'score_breakdown': agent_breakdown,
                'total_score': agent.get('baseline_score')
            })
        
        return {
            'query': user_query,
            'comparison': comparison_data,
            'methodology': {
                'description': 'Baseline scoring with interpretable features',
                'weights': self.scorer.weights
            }
        }


if __name__ == "__main__":
    # This would typically be imported and used with the data preprocessing module
    print("Baseline Agent Scorer initialized")
    print("Use with FeatureExtractor from data_preprocessing module")
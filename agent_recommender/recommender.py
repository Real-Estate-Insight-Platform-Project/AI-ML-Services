"""
Main Agent Recommender Interface.

This module provides a unified interface for both baseline and ML-based agent recommendations.
It handles model loading, query processing, and result formatting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from pathlib import Path
import json
import time

# Import local modules
from utils.data_preprocessing import AgentDataLoader, FeatureExtractor
from models.baseline_scorer import BaselineRecommender, BaselineAgentScorer
from models.ml_ranker import MLRecommender, MLAgentRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRecommenderSystem:
    """
    Unified Agent Recommender System supporting both baseline and ML approaches.
    """
    
    def __init__(self, data_path: str, model_path: Optional[str] = None):
        """
        Initialize the recommender system.
        
        Args:
            data_path: Path to the statewise agent data directory
            model_path: Path to saved ML model (optional)
        """
        self.data_path = Path(data_path)
        self.model_path = Path(model_path) if model_path else None
        
        # Initialize components
        self.data_loader = None
        self.feature_extractor = None
        self.baseline_recommender = None
        self.ml_recommender = None
        
        # Data
        self.agents_df = None
        
        # Status flags
        self.is_initialized = False
        self.ml_model_available = False
        
    def initialize(self, force_reload: bool = False):
        """
        Initialize the recommender system by loading data and models.
        
        Args:
            force_reload: Whether to force reload data even if already loaded
        """
        if self.is_initialized and not force_reload:
            logger.info("Recommender system already initialized")
            return
        
        logger.info("Initializing Agent Recommender System...")
        start_time = time.time()
        
        # Load and preprocess data
        logger.info("Loading agent data...")
        self.data_loader = AgentDataLoader(self.data_path)
        self.agents_df = self.data_loader.load_all_agents()
        self.agents_df = self.data_loader.preprocess_agents(self.agents_df)
        
        # Initialize feature extractor
        logger.info("Setting up feature extractor...")
        self.feature_extractor = FeatureExtractor(self.agents_df)
        
        # Initialize baseline recommender
        logger.info("Initializing baseline recommender...")
        self.baseline_recommender = BaselineRecommender(self.feature_extractor)
        
        # Initialize ML recommender
        logger.info("Initializing ML recommender...")
        ml_ranker = MLAgentRanker('lightgbm')
        self.ml_recommender = MLRecommender(self.feature_extractor, ml_ranker)
        
        # Load or train ML model
        if self.model_path and self.model_path.exists():
            try:
                logger.info(f"Loading ML model from {self.model_path}")
                self.ml_recommender.ml_ranker.load_model(str(self.model_path))
                self.ml_model_available = True
                logger.info("ML model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
                self.ml_model_available = False
        else:
            logger.info("No ML model found. Training new model...")
            try:
                self._train_ml_model()
            except Exception as e:
                logger.warning(f"Failed to train ML model: {e}")
                self.ml_model_available = False
        
        self.is_initialized = True
        elapsed_time = time.time() - start_time
        logger.info(f"Recommender system initialized in {elapsed_time:.2f} seconds")
        logger.info(f"Loaded {len(self.agents_df)} agents from {self.agents_df['state'].nunique()} states")
        logger.info(f"ML model available: {self.ml_model_available}")
    
    def _train_ml_model(self):
        """Train the ML model and optionally save it."""
        logger.info("Training ML model...")
        metrics = self.ml_recommender.train_model()
        self.ml_model_available = True
        
        logger.info(f"ML model training completed:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save model if path is provided
        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.ml_recommender.ml_ranker.save_model(str(self.model_path))
            logger.info(f"ML model saved to {self.model_path}")
    
    def recommend(self, user_query: Dict[str, Any], model_type: str = 'baseline',
                 top_k: int = 10, explain: bool = False) -> Dict[str, Any]:
        """
        Get agent recommendations for a user query.
        
        Args:
            user_query: User preferences dictionary containing:
                - regions: List[str] - target regions
                - budget: float - budget amount
                - property_types: List[str] - desired property types
            model_type: 'baseline', 'ml', or 'ensemble'
            top_k: Number of agents to recommend
            explain: Whether to include explanation
            
        Returns:
            dict: Recommendations with metadata
        """
        if not self.is_initialized:
            self.initialize()
        
        # Validate query
        validated_query = self._validate_user_query(user_query)
        
        if model_type == 'baseline':
            return self._get_baseline_recommendations(validated_query, top_k, explain)
        elif model_type == 'ml':
            return self._get_ml_recommendations(validated_query, top_k, explain)
        elif model_type == 'ensemble':
            return self._get_ensemble_recommendations(validated_query, top_k, explain)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    def _validate_user_query(self, user_query: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean user query."""
        validated = user_query.copy()
        
        # Ensure required fields exist
        validated.setdefault('regions', [])
        validated.setdefault('budget', 0)
        validated.setdefault('property_types', [])
        
        # Clean and validate regions
        if isinstance(validated['regions'], str):
            validated['regions'] = [validated['regions']]
        validated['regions'] = [r.strip() for r in validated['regions'] if r.strip()]
        
        # Validate budget
        try:
            validated['budget'] = float(validated['budget'])
            if validated['budget'] < 0:
                validated['budget'] = 0
        except (ValueError, TypeError):
            validated['budget'] = 0
        
        # Clean property types
        if isinstance(validated['property_types'], str):
            validated['property_types'] = [validated['property_types']]
        validated['property_types'] = [pt.strip() for pt in validated['property_types'] if pt.strip()]
        
        return validated
    
    def _get_baseline_recommendations(self, user_query: Dict[str, Any], 
                                   top_k: int, explain: bool) -> Dict[str, Any]:
        """Get recommendations from baseline model."""
        try:
            return self.baseline_recommender.recommend_agents(user_query, top_k, explain)
        except Exception as e:
            logger.error(f"Error in baseline recommendations: {e}")
            return {'error': str(e), 'model_type': 'baseline'}
    
    def _get_ml_recommendations(self, user_query: Dict[str, Any], 
                              top_k: int, explain: bool) -> Dict[str, Any]:
        """Get recommendations from ML model."""
        if not self.ml_model_available:
            return {
                'error': 'ML model not available. Using baseline instead.',
                'fallback_result': self._get_baseline_recommendations(user_query, top_k, explain)
            }
        
        try:
            return self.ml_recommender.recommend_agents(user_query, top_k, explain)
        except Exception as e:
            logger.error(f"Error in ML recommendations: {e}")
            return {
                'error': str(e),
                'model_type': 'ml',
                'fallback_result': self._get_baseline_recommendations(user_query, top_k, explain)
            }
    
    def _get_ensemble_recommendations(self, user_query: Dict[str, Any], 
                                    top_k: int, explain: bool) -> Dict[str, Any]:
        """Get ensemble recommendations combining baseline and ML."""
        baseline_result = self._get_baseline_recommendations(user_query, top_k * 2, False)
        
        if not self.ml_model_available:
            return {
                'warning': 'ML model not available. Using baseline only.',
                **baseline_result,
                'model_type': 'ensemble_baseline_only'
            }
        
        try:
            ml_result = self._get_ml_recommendations(user_query, top_k * 2, False)
            
            # Combine results with weighted scoring
            ensemble_result = self._combine_recommendations(
                baseline_result, ml_result, user_query, top_k, explain
            )
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error in ensemble recommendations: {e}")
            return {
                'warning': f'ML model error: {e}. Using baseline only.',
                **baseline_result,
                'model_type': 'ensemble_baseline_fallback'
            }
    
    def _combine_recommendations(self, baseline_result: Dict, ml_result: Dict,
                               user_query: Dict, top_k: int, explain: bool) -> Dict[str, Any]:
        """Combine baseline and ML recommendations."""
        # Extract agent scores
        baseline_agents = {
            rec['agentId']: rec for rec in baseline_result.get('recommendations', [])
        }
        ml_agents = {
            rec['agentId']: rec for rec in ml_result.get('recommendations', [])
        }
        
        # Combine scores with weights
        baseline_weight = 0.4
        ml_weight = 0.6
        
        combined_scores = {}
        all_agent_ids = set(baseline_agents.keys()).union(set(ml_agents.keys()))
        
        for agent_id in all_agent_ids:
            baseline_score = baseline_agents.get(agent_id, {}).get('score', 0)
            ml_score = ml_agents.get(agent_id, {}).get('score', 0)
            
            # Normalize scores to 0-1 range
            baseline_score_norm = baseline_score
            ml_score_norm = (ml_score - min(ml_result.get('recommendations', [{}])[i].get('score', 0) 
                                          for i in range(len(ml_result.get('recommendations', []))))) / \
                           (max(ml_result.get('recommendations', [{}])[i].get('score', 1) 
                               for i in range(len(ml_result.get('recommendations', [])))) - 
                            min(ml_result.get('recommendations', [{}])[i].get('score', 0) 
                               for i in range(len(ml_result.get('recommendations', [])))) + 1e-8)
            
            ensemble_score = baseline_weight * baseline_score_norm + ml_weight * ml_score_norm
            combined_scores[agent_id] = ensemble_score
        
        # Sort by ensemble score and take top-k
        top_agent_ids = sorted(combined_scores.keys(), 
                              key=lambda x: combined_scores[x], reverse=True)[:top_k]
        
        # Build recommendations
        recommendations = []
        for agent_id in top_agent_ids:
            # Use ML agent data if available, otherwise baseline
            agent_data = ml_agents.get(agent_id, baseline_agents.get(agent_id, {}))
            if agent_data:
                agent_data = agent_data.copy()
                agent_data['score'] = combined_scores[agent_id]
                agent_data['ensemble_components'] = {
                    'baseline_score': baseline_agents.get(agent_id, {}).get('score', 0),
                    'ml_score': ml_agents.get(agent_id, {}).get('score', 0)
                }
                recommendations.append(agent_data)
        
        result = {
            'query': user_query,
            'recommendations': recommendations,
            'total_agents_evaluated': max(
                baseline_result.get('total_agents_evaluated', 0),
                ml_result.get('total_agents_evaluated', 0)
            ),
            'model_type': 'ensemble',
            'ensemble_weights': {
                'baseline': baseline_weight,
                'ml': ml_weight
            }
        }
        
        if explain:
            result['explanation'] = {
                'ensemble_method': 'Weighted combination of baseline and ML scores',
                'weights': {'baseline': baseline_weight, 'ml': ml_weight},
                'baseline_explanation': baseline_result.get('explanation'),
                'ml_explanation': ml_result.get('explanation')
            }
        
        return result
    
    def compare_models(self, user_query: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        """
        Compare recommendations from different models.
        
        Args:
            user_query: User preferences dictionary
            top_k: Number of agents to compare
            
        Returns:
            dict: Comparison results
        """
        if not self.is_initialized:
            self.initialize()
        
        validated_query = self._validate_user_query(user_query)
        
        baseline_result = self._get_baseline_recommendations(validated_query, top_k, True)
        
        comparison = {
            'query': validated_query,
            'baseline': baseline_result,
            'models_compared': ['baseline']
        }
        
        if self.ml_model_available:
            ml_result = self._get_ml_recommendations(validated_query, top_k, True)
            comparison['ml'] = ml_result
            comparison['models_compared'].append('ml')
            
            # Calculate overlap
            baseline_ids = {rec['agentId'] for rec in baseline_result.get('recommendations', [])}
            ml_ids = {rec['agentId'] for rec in ml_result.get('recommendations', [])}
            
            overlap = len(baseline_ids.intersection(ml_ids))
            comparison['overlap_analysis'] = {
                'agents_in_common': overlap,
                'overlap_percentage': (overlap / top_k) * 100 if top_k > 0 else 0,
                'baseline_unique': len(baseline_ids - ml_ids),
                'ml_unique': len(ml_ids - baseline_ids)
            }
        
        return comparison
    
    def get_agent_details(self, agent_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            dict: Agent details
        """
        if not self.is_initialized:
            self.initialize()
        
        agent_data = self.agents_df[self.agents_df['agentId'] == agent_id]
        
        if len(agent_data) == 0:
            return {'error': f'Agent {agent_id} not found'}
        
        agent = agent_data.iloc[0]
        
        details = {
            'agentId': agent.get('agentId'),
            'name': agent.get('name'),
            'email': agent.get('email'),
            'phoneNumber': agent.get('phoneNumber'),
            'starRating': agent.get('starRating'),
            'numReviews': agent.get('numReviews'),
            'pastYearDeals': agent.get('pastYearDeals'),
            'homeTransactionsLifetime': agent.get('homeTransactionsLifetime'),
            'transactionVolumeLifetime': agent.get('transactionVolumeLifetime'),
            'businessMarket': agent.get('businessMarket'),
            'brokerageName': agent.get('brokerageName'),
            'primaryServiceRegions': agent.get('primaryServiceRegions'),
            'propertyTypes': agent.get('propertyTypes'),
            'partner': agent.get('partner'),
            'isPremier': agent.get('isPremier'),
            'state': agent.get('state'),
            'jobTitle': agent.get('jobTitle'),
            'profileUrl': agent.get('profileUrl'),
            'photoUrl': agent.get('photoUrl'),
            'statistics': {
                'avg_transaction_value': agent.get('avg_transaction_value', 0),
                'deal_price_median': agent.get('dealPrices_median', 0),
                'deal_price_range': {
                    'min': agent.get('dealPrices_min', 0),
                    'max': agent.get('dealPrices_max', 0),
                    'q25': agent.get('dealPrices_q25', 0),
                    'q75': agent.get('dealPrices_q75', 0)
                }
            }
        }
        
        return details
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        if not self.is_initialized:
            self.initialize()
        
        stats = {
            'total_agents': len(self.agents_df),
            'unique_states': self.agents_df['state'].nunique(),
            'unique_markets': self.agents_df['businessMarket'].nunique(),
            'unique_brokerages': self.agents_df['brokerageName'].nunique(),
            'models_available': {
                'baseline': True,
                'ml': self.ml_model_available
            },
            'data_summary': {
                'avg_star_rating': self.agents_df['starRating'].mean(),
                'avg_reviews': self.agents_df['numReviews'].mean(),
                'avg_past_year_deals': self.agents_df['pastYearDeals'].mean(),
                'total_transaction_volume': self.agents_df['transactionVolumeLifetime'].sum()
            }
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    data_path = "../agent_data/statewise_data"
    model_path = "saved_models/agent_ranker_model.joblib"
    
    # Initialize system
    recommender = AgentRecommenderSystem(data_path, model_path)
    recommender.initialize()
    
    # Example query
    user_query = {
        'regions': ['Fort Myers', 'Naples'],
        'budget': 500000,
        'property_types': ['Single Family Residential']
    }
    
    # Get recommendations
    baseline_recs = recommender.recommend(user_query, model_type='baseline', top_k=5, explain=True)
    print(f"Baseline recommendations: {len(baseline_recs.get('recommendations', []))}")
    
    if recommender.ml_model_available:
        ml_recs = recommender.recommend(user_query, model_type='ml', top_k=5, explain=True)
        print(f"ML recommendations: {len(ml_recs.get('recommendations', []))}")
    
    # Compare models
    comparison = recommender.compare_models(user_query, top_k=5)
    print(f"Models compared: {comparison['models_compared']}")
    
    # System stats
    stats = recommender.get_system_stats()
    print(f"System loaded {stats['total_agents']} agents from {stats['unique_states']} states")
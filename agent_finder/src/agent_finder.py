"""
Main Agent Finder orchestrator that coordinates the entire ML pipeline.

Orchestrates:
- Data processing and cleaning
- Theme mining and embeddings
- Skill vector construction and calibration
- Weight learning and preference fusion
- Agent recommendation and ranking
- Explanation generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
import os
from datetime import datetime

from .data_processor import DataProcessor
from .theme_miner import AdvancedThemeMiner
from .skill_vectors import SkillVectorBuilder
from .weight_learner import WeightLearner
from .recommendation_engine import RecommendationEngine
from .explanations import ExplanationGenerator

logger = logging.getLogger(__name__)

class AgentFinder:
    """Main orchestrator for the agent recommendation system."""
    
    def __init__(self,
                 agents_file: str,
                 reviews_file: str,
                 cache_dir: str = "./cache",
                 model_params: Optional[Dict] = None):
        """
        Initialize Agent Finder system.
        
        Args:
            agents_file: Path to agents CSV file
            reviews_file: Path to reviews CSV file  
            cache_dir: Directory for caching processed data and models
            model_params: Optional parameters for model components
        """
        self.agents_file = agents_file
        self.reviews_file = reviews_file
        self.cache_dir = cache_dir
        self.model_params = model_params or {}
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor(agents_file, reviews_file)
        self.theme_miner = AdvancedThemeMiner(**self.model_params.get('theme_miner', {}))
        self.skill_builder = SkillVectorBuilder(**self.model_params.get('skill_builder', {}))
        self.weight_learner = WeightLearner(**self.model_params.get('weight_learner', {}))
        self.recommender = RecommendationEngine(**self.model_params.get('recommender', {}))
        
        # Processed data and models (loaded lazily)
        self.agents_df = None
        self.reviews_df = None
        self.agent_theme_matrix = None
        self.theme_results = None
        self.agents_skill_df = None
        self.base_weights = None
        self.explanation_generator = None
        
        # Training status
        self.is_trained = False
        self.training_timestamp = None
    
    def train_system(self, use_cache: bool = True, save_cache: bool = True) -> Dict[str, Any]:
        """
        Train the complete agent recommendation system.
        
        Args:
            use_cache: Whether to use cached intermediate results
            save_cache: Whether to save results to cache
            
        Returns:
            Training summary with metrics and statistics
        """
        logger.info("Starting Agent Finder system training...")
        start_time = datetime.now()
        
        training_summary = {
            'start_time': start_time,
            'steps_completed': [],
            'metrics': {}
        }
        
        try:
            # Step 1: Data Processing
            logger.info("Step 1: Data Processing and Cleaning")
            cache_file = os.path.join(self.cache_dir, "processed_data.pkl")
            
            if use_cache and os.path.exists(cache_file):
                logger.info("Loading cached processed data...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.agents_df = cached_data['agents_df']
                    self.reviews_df = cached_data['reviews_df']
            else:
                self.agents_df, self.reviews_df = self.data_processor.process_all_data()
                
                # Add learned quality prior (replaces hardcoded Q_prior blend)
                logger.info("Step 1b: Learning Q_prior blend weights from data...")
                self.agents_df = self.data_processor.calculate_learned_quality_prior(self.agents_df)
                
                if save_cache:
                    with open(cache_file, 'wb') as f:
                        pickle.dump({
                            'agents_df': self.agents_df,
                            'reviews_df': self.reviews_df
                        }, f)
            
            training_summary['steps_completed'].append('data_processing')
            training_summary['metrics']['n_agents'] = len(self.agents_df)
            training_summary['metrics']['n_reviews'] = len(self.reviews_df)
            
            # Step 2: Theme Mining
            logger.info("Step 2: Theme Mining and Embeddings")
            theme_cache_file = os.path.join(self.cache_dir, "theme_results.pkl")
            
            if use_cache and os.path.exists(theme_cache_file):
                logger.info("Loading cached theme mining results...")
                with open(theme_cache_file, 'rb') as f:
                    cached_themes = pickle.load(f)
                    self.agent_theme_matrix = cached_themes['agent_theme_matrix']
                    self.theme_results = cached_themes['theme_results']
            else:
                self.agent_theme_matrix, self.theme_results = self.theme_miner.fit_transform(
                    self.reviews_df, self.agents_df
                )
                
                if save_cache:
                    with open(theme_cache_file, 'wb') as f:
                        pickle.dump({
                            'agent_theme_matrix': self.agent_theme_matrix,
                            'theme_results': self.theme_results
                        }, f)
            
            training_summary['steps_completed'].append('theme_mining')
            training_summary['metrics']['n_themes'] = self.theme_results.get('n_themes', 0)
            training_summary['metrics']['noise_ratio'] = self.theme_results.get('noise_ratio', 0.0)
            
            # Step 3: Skill Vector Construction
            logger.info("Step 3: Skill Vector Construction and Calibration")
            self.agents_skill_df = self.skill_builder.build_agent_skill_vectors(
                self.agents_df, self.agent_theme_matrix
            )
            
            # Add confidence metrics
            self.agents_skill_df = self.skill_builder.calculate_confidence_metrics(self.agents_skill_df)
            
            training_summary['steps_completed'].append('skill_vectors')
            
            # Step 4: Weight Learning
            logger.info("Step 4: Learning Base Weights from Data")
            weight_cache_file = os.path.join(self.cache_dir, "learned_weights.pkl")
            
            if use_cache and os.path.exists(weight_cache_file):
                logger.info("Loading cached learned weights...")
                with open(weight_cache_file, 'rb') as f:
                    cached_weights = pickle.load(f)
                    self.weight_learner = cached_weights['weight_learner']
                    self.base_weights = cached_weights['base_weights']
            else:
                # Prepare training data
                X, y, feature_names = self.weight_learner.prepare_training_data(
                    self.reviews_df, self.agents_skill_df, self.theme_results
                )
                
                # Learn base weights
                self.base_weights = self.weight_learner.fit_base_weights(X, y)
                
                if save_cache:
                    with open(weight_cache_file, 'wb') as f:
                        pickle.dump({
                            'weight_learner': self.weight_learner,
                            'base_weights': self.base_weights
                        }, f)
            
            training_summary['steps_completed'].append('weight_learning')
            training_summary['metrics']['model_performance'] = self.weight_learner.model_metrics
            
            # Step 5: Initialize Explanation Generator
            logger.info("Step 5: Initializing Explanation System")
            self.explanation_generator = ExplanationGenerator(
                self.theme_results, 
                self.weight_learner.feature_names
            )
            
            training_summary['steps_completed'].append('explanation_system')
            
            # Mark as trained
            self.is_trained = True
            self.training_timestamp = datetime.now()
            
            # Final summary
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            training_summary.update({
                'end_time': end_time,
                'training_time_seconds': training_time,
                'success': True,
                'is_trained': self.is_trained
            })
            
            logger.info(f"Agent Finder training completed successfully in {training_time:.1f} seconds")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            training_summary.update({
                'end_time': datetime.now(),
                'success': False,
                'error': str(e)
            })
            raise
    
    def recommend_agents(self,
                        user_preferences: Dict[str, float],
                        user_filters: Dict[str, Any],
                        top_k: int = 10,
                        include_explanations: bool = True) -> Dict[str, Any]:
        """
        Generate personalized agent recommendations.
        
        Args:
            user_preferences: User preference sliders (0-1 values)
            user_filters: Hard filters (location, price, etc.)
            top_k: Number of recommendations to return
            include_explanations: Whether to include detailed explanations
            
        Returns:
            Complete recommendation results with rankings and explanations
        """
        if not self.is_trained:
            raise ValueError("System must be trained before generating recommendations")
        
        logger.info(f"Generating {top_k} agent recommendations...")
        
        # Step 1: Fuse preferences with learned weights
        fused_weights, alpha = self.weight_learner.fuse_preferences(user_preferences)
        
        # Step 2: Generate recommendations
        theme_columns = [col for col in self.agent_theme_matrix.columns if col.startswith('theme_')]
        
        # Store user preferences in recommender for direct personalization
        self.recommender._current_user_preferences = user_preferences
        
        recommended_agents, metadata = self.recommender.rank_and_recommend(
            self.agents_skill_df,
            fused_weights,
            self.weight_learner.feature_names,
            theme_columns,
            user_filters,
            top_k
        )
        
        # Step 3: Generate explanations if requested
        explanations = []
        if include_explanations and len(recommended_agents) > 0:
            explanations = self.explanation_generator.generate_agent_explanations(
                recommended_agents,
                self.reviews_df,
                user_preferences,
                fused_weights,
                metadata
            )
        
        # Step 4: Prepare final results
        results = {
            'recommendations': self._format_recommendations(recommended_agents),
            'explanations': explanations,
            'metadata': {
                **metadata,
                'fusion_alpha': alpha,
                'user_preferences': user_preferences,
                'user_filters': user_filters,
                'recommendation_timestamp': datetime.now().isoformat(),
                'system_trained_at': self.training_timestamp.isoformat() if self.training_timestamp else None
            },
            'summary': {
                'total_candidates': metadata.get('total_agents', 0),
                'after_filtering': metadata.get('after_constraint_filter', 0),
                'recommended': len(recommended_agents),
                'preference_personalization': f"{alpha:.2f}" if alpha else "0.00"
            }
        }
        
        logger.info(f"Generated {len(recommended_agents)} recommendations successfully")
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the trained system."""
        if not self.is_trained:
            return {'is_trained': False}
        
        stats = {
            'is_trained': True,
            'training_timestamp': self.training_timestamp.isoformat() if self.training_timestamp else None,
            'data_stats': {
                'n_agents': len(self.agents_df) if self.agents_df is not None else 0,
                'n_reviews': len(self.reviews_df) if self.reviews_df is not None else 0,
                'agents_with_reviews': len(self.agents_df[self.agents_df['review_count_actual'] > 0]) if self.agents_df is not None else 0
            },
            'theme_stats': {
                'n_themes': self.theme_results.get('n_themes', 0) if self.theme_results else 0,
                'noise_ratio': self.theme_results.get('noise_ratio', 0.0) if self.theme_results else 0.0,
                'theme_names': list(self.theme_results.get('theme_labels', {}).values()) if self.theme_results else []
            },
            'model_stats': {
                'n_features': len(self.weight_learner.feature_names) if self.weight_learner.feature_names else 0,
                'model_performance': self.weight_learner.model_metrics if hasattr(self.weight_learner, 'model_metrics') else {},
                'top_feature_weights': self._get_top_feature_weights()
            }
        }
        
        return stats
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        """Safely convert value to float, handling NaN and infinity."""
        try:
            if pd.isna(value) or np.isinf(value):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _format_recommendations(self, recommended_agents: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format agent recommendations for API response."""
        if len(recommended_agents) == 0:
            return []
        
        recommendations = []
        
        for _, agent in recommended_agents.iterrows():
            rec = {
                'agent_id': int(agent['advertiser_id']),
                'name': agent.get('full_name', 'Unknown'),
                'rank': int(agent.get('rank', 0)),
                'utility_score': self._safe_float(agent.get('utility_score', 0.0)),
                'availability_fit': self._safe_float(agent.get('availability_fit', 0.0)),
                'confidence_score': self._safe_float(agent.get('confidence_score', 0.0)),
                'profile': {
                    'state': agent.get('state', ''),
                    'city': agent.get('agent_base_city', ''),
                    'rating': self._safe_float(agent.get('agent_rating', 0.0)),
                    'review_count': int(agent.get('review_count_actual', 0)),
                    'experience_years': self._safe_float(agent.get('experience_years', 0.0)),
                    'specializations': agent.get('specializations', ''),
                    'languages': agent.get('languages', 'English'),
                    'agent_type': str(agent.get('agent_type', '')) if pd.notna(agent.get('agent_type')) else '',
                    'office_name': str(agent.get('office_name', '')) if pd.notna(agent.get('office_name')) else '',
                    'phone': str(agent.get('phone_primary', '')) if pd.notna(agent.get('phone_primary')) else '',
                    'website': str(agent.get('agent_website', '')) if pd.notna(agent.get('agent_website')) else '',
                    'bio': (str(agent.get('agent_bio', ''))[:200] + '...' if pd.notna(agent.get('agent_bio')) and len(str(agent.get('agent_bio', ''))) > 200 else str(agent.get('agent_bio', ''))) if pd.notna(agent.get('agent_bio')) else '',
                },
                'metrics': {
                    'responsiveness': self._safe_float(agent.get('skill_responsiveness', 0.0)),
                    'negotiation': self._safe_float(agent.get('skill_negotiation', 0.0)),
                    'professionalism': self._safe_float(agent.get('skill_professionalism', 0.0)),
                    'market_expertise': self._safe_float(agent.get('skill_market_expertise', 0.0)),
                    'q_prior': self._safe_float(agent.get('q_prior', 0.0)),
                    'wilson_lower_bound': self._safe_float(agent.get('wilson_lower_bound', 0.0)),
                    'recency_score': self._safe_float(agent.get('recency_score', 0.0))
                }
            }
            
            recommendations.append(rec)
        
        return recommendations
    
    def _get_top_feature_weights(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top feature weights for system stats."""
        if not self.base_weights:
            return []
        
        sorted_weights = sorted(
            self.base_weights.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return sorted_weights[:top_k]
    
    def save_model(self, filepath: str) -> None:
        """Save the complete trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'agents_df': self.agents_df,
            'reviews_df': self.reviews_df,
            'agent_theme_matrix': self.agent_theme_matrix,
            'theme_results': self.theme_results,
            'agents_skill_df': self.agents_skill_df,
            'weight_learner': self.weight_learner,
            'base_weights': self.base_weights,
            'skill_builder': self.skill_builder,
            'recommender': self.recommender,
            'training_timestamp': self.training_timestamp,
            'model_params': self.model_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'AgentFinder':
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            agents_file="",  # Not needed for loaded model
            reviews_file="",
            model_params=model_data.get('model_params', {})
        )
        
        # Restore state
        instance.agents_df = model_data['agents_df']
        instance.reviews_df = model_data['reviews_df']
        instance.agent_theme_matrix = model_data['agent_theme_matrix']
        instance.theme_results = model_data['theme_results']
        instance.agents_skill_df = model_data['agents_skill_df']
        instance.weight_learner = model_data['weight_learner']
        instance.base_weights = model_data['base_weights']
        instance.skill_builder = model_data['skill_builder']
        instance.recommender = model_data['recommender']
        instance.training_timestamp = model_data['training_timestamp']
        
        # Initialize explanation generator
        instance.explanation_generator = ExplanationGenerator(
            instance.theme_results,
            instance.weight_learner.feature_names
        )
        
        instance.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance
"""
Data-driven weight learning and preference fusion system.

Implements:
- Ridge regression to learn base importance weights from review data
- User preference vector construction from slider inputs  
- Adaptive preference fusion with learned base weights
- Cross-validation for hyperparameter optimization
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class WeightLearner:
    """Learns data-driven importance weights and handles preference fusion."""
    
    def __init__(self, 
                 model_type: str = 'ridge',
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize weight learning system.
        
        Args:
            model_type: Type of regression model ('ridge' or 'elasticnet')
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Fitted model and parameters
        self.model = None
        self.base_weights = None
        self.feature_names = None
        self.model_metrics = {}
        
        # Preference system parameters
        self.preference_mapping = {
            'responsiveness': 'skill_responsiveness',
            'negotiation': 'skill_negotiation', 
            'professionalism': 'skill_professionalism',
            'market_expertise': 'skill_market_expertise'
        }
    
    def prepare_training_data(self,
                            reviews_df: pd.DataFrame,
                            agents_skill_df: pd.DataFrame,
                            theme_results: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data at the review level for weight learning.
        
        Args:
            reviews_df: Review data with ratings and metadata
            agents_skill_df: Agent skill vectors and features
            theme_results: Results from theme mining including cluster labels
            
        Returns:
            X: Feature matrix (reviews × features)
            y: Target ratings
            feature_names: List of feature names
        """
        logger.info("Preparing training data for weight learning...")
        
        # Add theme clusters to reviews
        reviews_with_themes = reviews_df.copy()
        reviews_with_themes['theme_cluster'] = theme_results['cluster_labels']
        
        # Remove reviews without ratings or agent matches
        valid_reviews = reviews_with_themes[
            (reviews_with_themes['review_rating'].notna()) &
            (reviews_with_themes['advertiser_id'].isin(agents_skill_df['advertiser_id']))
        ].copy()
        
        logger.info(f"Using {len(valid_reviews)} valid reviews for training")
        
        # Prepare feature matrix
        features = []
        feature_names = []
        
        for _, review in valid_reviews.iterrows():
            agent_id = review['advertiser_id']
            agent_data = agents_skill_df[agents_skill_df['advertiser_id'] == agent_id].iloc[0]
            
            # Sub-scores (if available in review)
            review_features = {}
            
            # Use review-level sub-scores if available, otherwise agent averages
            for subscore in ['sub_responsiveness', 'sub_negotiation', 'sub_professionalism', 'sub_market_expertise']:
                if pd.notna(review[subscore]):
                    review_features[subscore] = review[subscore]
                else:
                    # Use agent average (skill vector)
                    skill_col = f"skill_{subscore.replace('sub_', '')}"
                    review_features[subscore] = agent_data.get(skill_col, 0.0)
            
            # Agent theme strengths
            theme_cols = [col for col in agents_skill_df.columns if col.startswith('theme_')]
            for theme_col in theme_cols:
                review_features[theme_col] = agent_data.get(theme_col, 0.0)
            
            # Context features
            review_features['is_buyer_review'] = float(review.get('is_buyer_review', False))
            review_features['is_seller_review'] = float(review.get('is_seller_review', False))
            review_features['recency_weight'] = review.get('recency_weight', 1.0)
            
            # Agent stability features
            review_features['volume_score'] = agent_data.get('volume_score', 0.0)
            review_features['experience_score'] = agent_data.get('experience_score', 0.0)
            
            features.append(list(review_features.values()))
            
            # Store feature names (only once)
            if not feature_names:
                feature_names = list(review_features.keys())
        
        # Convert to arrays
        X = np.array(features)
        y = valid_reviews['review_rating'].values
        
        # Handle NaN values by imputation
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Store imputer and feature names for later use
        self.imputer = imputer
        self.feature_names = feature_names
        
        logger.info(f"Created feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        logger.info(f"Handled NaN values using median imputation")
        
        return X, y, feature_names
    
    def fit_base_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Fit regression model to learn base importance weights.
        
        Args:
            X: Feature matrix
            y: Target ratings
            
        Returns:
            Dictionary of base weights
        """
        logger.info(f"Learning base weights using {self.model_type} regression...")
        
        # Hyperparameter search
        if self.model_type == 'ridge':
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
            base_model = Ridge(random_state=self.random_state)
        elif self.model_type == 'elasticnet':
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            }
            base_model = ElasticNet(random_state=self.random_state, max_iter=2000)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        # Extract coefficients as base weights
        coefficients = self.model.coef_
        self.base_weights = dict(zip(self.feature_names, coefficients))
        
        # Calculate performance metrics
        y_pred = self.model.predict(X)
        self.model_metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'best_params': grid_search.best_params_,
            'cv_score': -grid_search.best_score_
        }
        
        logger.info(f"Model performance - R²: {self.model_metrics['r2']:.3f}, RMSE: {self.model_metrics['rmse']:.3f}")
        logger.info(f"Best parameters: {self.model_metrics['best_params']}")
        
        # Log top feature importances
        sorted_weights = sorted(self.base_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        logger.info("Top 10 feature importances:")
        for feature, weight in sorted_weights[:10]:
            logger.info(f"  {feature}: {weight:.4f}")
        
        return self.base_weights
    
    def create_preference_vector(self, user_preferences: Dict[str, float]) -> np.ndarray:
        """
        Convert user slider inputs to preference vector.
        
        Args:
            user_preferences: Dict with preference values (e.g., {'responsiveness': 0.8, 'negotiation': 0.6})
            
        Returns:
            Preference vector aligned with feature names
        """
        if self.feature_names is None:
            raise ValueError("Must fit model first to get feature names")
        
        # Initialize preference vector with zeros
        preference_vector = np.zeros(len(self.feature_names))
        
        # Map user preferences to feature indices
        for pref_name, pref_value in user_preferences.items():
            if pref_name in self.preference_mapping:
                feature_name = self.preference_mapping[pref_name]
                if feature_name in self.feature_names:
                    feature_idx = self.feature_names.index(feature_name)
                    # Scale preference values to have more impact (multiply by 10)
                    preference_vector[feature_idx] = pref_value * 10.0
        
        return preference_vector
    
    def fuse_preferences(self,
                        user_preferences: Dict[str, float],
                        normalize: bool = True) -> Tuple[np.ndarray, float]:
        """
        Fuse learned base weights with user preferences.
        
        Args:
            user_preferences: User preference values
            normalize: Whether to normalize the final weights
            
        Returns:
            Fused weights vector and alpha (fusion strength)
        """
        if self.base_weights is None:
            raise ValueError("Must fit base weights first")
        
        # Convert base weights to vector
        base_weight_vector = np.array([self.base_weights[name] for name in self.feature_names])
        
        # Create preference vector
        preference_vector = self.create_preference_vector(user_preferences)
        
        # Calculate alpha (fusion strength) based on how much user moved sliders
        # Assume neutral is 0.5, so deviation from neutral indicates preference strength
        neutral_value = 0.5
        preference_deviations = [abs(v - neutral_value) for v in user_preferences.values()]
        alpha = np.mean(preference_deviations) * 2  # Scale to 0-1 range
        alpha = np.clip(alpha, 0.0, 1.0)
        
        # Fuse weights: (1-α) * base + α * preferences
        fused_weights = (1 - alpha) * base_weight_vector + alpha * preference_vector
        
        # Normalize if requested
        if normalize:
            # Use L2 normalization to preserve relative magnitudes
            norm = np.linalg.norm(fused_weights)
            if norm > 0:
                fused_weights = fused_weights / norm
        
        logger.info(f"Fused preferences with α = {alpha:.3f}")
        
        return fused_weights, alpha
    
    def get_feature_importance_explanation(self, top_k: int = 10) -> List[Tuple[str, float, str]]:
        """
        Get human-readable explanations for top feature importances.
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            List of tuples (feature_name, importance, explanation)
        """
        if self.base_weights is None:
            return []
        
        # Sort by absolute importance
        sorted_weights = sorted(self.base_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Generate explanations
        explanations = []
        for feature, weight in sorted_weights[:top_k]:
            explanation = self._generate_feature_explanation(feature, weight)
            explanations.append((feature, weight, explanation))
        
        return explanations
    
    def _generate_feature_explanation(self, feature_name: str, weight: float) -> str:
        """Generate human-readable explanation for a feature."""
        
        # Feature explanation mapping
        explanations = {
            'sub_responsiveness': 'Agent responsiveness to communication',
            'sub_negotiation': 'Agent negotiation skills', 
            'sub_professionalism': 'Agent professionalism and conduct',
            'sub_market_expertise': 'Agent market knowledge and expertise',
            'skill_responsiveness': 'Agent responsiveness (from reviews)',
            'skill_negotiation': 'Agent negotiation ability (from reviews)',
            'skill_professionalism': 'Agent professionalism (from reviews)', 
            'skill_market_expertise': 'Agent market expertise (from reviews)',
            'is_buyer_review': 'Review from buyer vs seller',
            'is_seller_review': 'Review from seller vs buyer',
            'recency_weight': 'How recent the review is',
            'volume_score': 'Agent review volume',
            'experience_score': 'Agent years of experience'
        }
        
        # Theme explanations
        if feature_name.startswith('theme_'):
            theme_name = feature_name.replace('theme_', '').replace('_', ' ').title()
            base_explanation = f'Strength in "{theme_name}" theme'
        else:
            base_explanation = explanations.get(feature_name, feature_name.replace('_', ' ').title())
        
        # Add direction indicator
        direction = "positively" if weight > 0 else "negatively"
        strength = "strongly" if abs(weight) > 0.1 else "moderately"
        
        return f"{base_explanation} - {strength} {direction} affects rating"
    
    def predict_rating(self, agent_features: np.ndarray) -> float:
        """Predict rating for agent features using fitted model."""
        if self.model is None:
            raise ValueError("Must fit model first")
        
        # Apply same imputation as during training
        if hasattr(self, 'imputer'):
            agent_features = self.imputer.transform(agent_features.reshape(1, -1))
        else:
            agent_features = agent_features.reshape(1, -1)
        
        return self.model.predict(agent_features)[0]
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of fitted model."""
        return {
            'model_type': self.model_type,
            'metrics': self.model_metrics,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'base_weights': self.base_weights
        }
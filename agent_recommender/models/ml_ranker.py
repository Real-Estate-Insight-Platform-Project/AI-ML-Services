"""
Machine Learning-based Agent Recommender.

This module implements advanced ML models (LightGBM, XGBoost) for learning
agent ranking patterns from data, complementing the baseline interpretable model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import ndcg_score, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import joblib
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLFeatureEngineer:
    """Advanced feature engineering for ML models."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_features = [
            'businessMarket', 'brokerageName', 'jobTitle', 'state', 'officeState'
        ]
        
    def engineer_ml_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for ML models.
        
        Args:
            features_df: Base features DataFrame
            
        Returns:
            pandas.DataFrame: Enhanced features for ML
        """
        df = features_df.copy()
        
        # Interaction features
        df['rating_x_reviews'] = df['starRating'] * np.log1p(df['numReviews'])
        df['volume_per_transaction'] = np.where(
            df['homeTransactionsLifetime'] > 0,
            df['transactionVolumeLifetime'] / df['homeTransactionsLifetime'],
            0
        )
        df['deals_recency_ratio'] = np.where(
            df['homeTransactionsLifetime'] > 0,
            df['pastYearDeals'] / df['homeTransactionsLifetime'],
            0
        )
        
        # Price range features
        df['price_range'] = df['dealPrices_max'] - df['dealPrices_min']
        df['price_coefficient_variation'] = np.where(
            df['dealPrices_median'] > 0,
            df['dealPrices_std'] / df['dealPrices_median'],
            0
        )
        
        # Experience and specialization features
        df['is_specialist'] = (df['num_property_types'] <= 3).astype(int)
        df['is_regional_expert'] = (df['num_service_regions'] <= 5).astype(int)
        df['experience_score'] = (
            np.log1p(df['homeTransactionsLifetime']) * 0.7 +
            np.log1p(df['transactionVolumeLifetime']) * 0.3
        )
        
        # Market position features
        df['high_volume_agent'] = (df['pastYearDeals'] >= df['pastYearDeals'].quantile(0.8)).astype(int)
        df['premium_agent'] = ((df['starRating'] >= 4.5) & (df['numReviews'] >= 10)).astype(int)
        
        # Availability and service features
        df['full_service'] = (df['servesOffers'] & df['servesListings']).astype(int)
        df['availability_score'] = (
            df['isActive'].astype(float) * 0.4 +
            df['profileContactEnabled'].astype(float) * 0.3 +
            df['full_service'] * 0.3
        )
        
        # Handle categorical features
        for cat_col in self.categorical_features:
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].fillna('Unknown')
                
        return df
    
    def prepare_ml_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for ML training/prediction.
        
        Args:
            df: Feature DataFrame
            fit: Whether to fit encoders/scalers
            
        Returns:
            Tuple of (processed_df, feature_names)
        """
        df = df.copy()
        
        # Encode categorical features
        for cat_col in self.categorical_features:
            if cat_col in df.columns:
                if fit:
                    if cat_col not in self.label_encoders:
                        self.label_encoders[cat_col] = LabelEncoder()
                    df[f'{cat_col}_encoded'] = self.label_encoders[cat_col].fit_transform(df[cat_col])
                else:
                    if cat_col in self.label_encoders:
                        # Handle unseen categories
                        known_labels = set(self.label_encoders[cat_col].classes_)
                        df[cat_col] = df[cat_col].apply(
                            lambda x: x if x in known_labels else 'Unknown'
                        )
                        df[f'{cat_col}_encoded'] = self.label_encoders[cat_col].transform(df[cat_col])
                    else:
                        df[f'{cat_col}_encoded'] = 0
        
        # Select ML features
        ml_features = [
            # Base scoring features
            'geo_overlap', 'price_band_match', 'property_type_match',
            'normalized_recency', 'rating_score', 'log_reviews_normalized',
            'partner_premier_boost',
            
            # Agent characteristics
            'starRating', 'numReviews', 'pastYearDeals', 'pastYearDealsInRegion',
            'homeTransactionsLifetime', 'transactionVolumeLifetime',
            'dealPrices_median', 'dealPrices_q25', 'dealPrices_q75',
            'dealPrices_min', 'dealPrices_max', 'dealPrices_std',
            'num_service_regions', 'num_property_types',
            
            # Engineered features
            'rating_x_reviews', 'volume_per_transaction', 'deals_recency_ratio',
            'price_range', 'price_coefficient_variation', 'is_specialist',
            'is_regional_expert', 'experience_score', 'high_volume_agent',
            'premium_agent', 'availability_score',
            
            # Boolean flags
            'partner', 'isPremier', 'servesOffers', 'servesListings',
            'isActive', 'profileContactEnabled'
        ]
        
        # Add encoded categorical features
        for cat_col in self.categorical_features:
            if f'{cat_col}_encoded' in df.columns:
                ml_features.append(f'{cat_col}_encoded')
        
        # Keep only features that exist in DataFrame
        available_features = [f for f in ml_features if f in df.columns]
        
        # Fill missing values
        feature_df = df[available_features].fillna(0)
        
        return feature_df, available_features


class MLAgentRanker:
    """Machine Learning-based agent ranking model."""
    
    def __init__(self, model_type: str = 'lightgbm'):
        """
        Initialize ML ranker.
        
        Args:
            model_type: Type of model ('lightgbm', 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.feature_engineer = MLFeatureEngineer()
        self.feature_names = None
        self.is_trained = False
        
    def _create_synthetic_labels(self, features_df: pd.DataFrame, 
                                baseline_scores: pd.Series) -> pd.Series:
        """
        Create synthetic training labels from baseline scores and additional signals.
        
        Args:
            features_df: Features DataFrame
            baseline_scores: Baseline model scores
            
        Returns:
            pandas.Series: Synthetic labels for training
        """
        # Start with baseline scores
        labels = baseline_scores.copy()
        
        # Add noise and business logic adjustments
        
        # Boost agents with very high ratings and many reviews
        high_quality_mask = (
            (features_df['starRating'] >= 4.8) & 
            (features_df['numReviews'] >= 20)
        )
        labels[high_quality_mask] += 0.1
        
        # Boost very active agents
        high_activity_mask = features_df['pastYearDeals'] >= features_df['pastYearDeals'].quantile(0.9)
        labels[high_activity_mask] += 0.05
        
        # Slight penalty for agents with no recent activity
        no_activity_mask = features_df['pastYearDeals'] == 0
        labels[no_activity_mask] -= 0.1
        
        # Boost agents with good price range coverage
        good_coverage_mask = features_df['dealPrices_count'] >= 10
        labels[good_coverage_mask] += 0.03
        
        # Add some random noise for diversity
        noise = np.random.normal(0, 0.02, len(labels))
        labels += noise
        
        # Clip to [0, 1] range
        labels = np.clip(labels, 0, 1)
        
        return labels
    
    def prepare_training_data(self, features_df: pd.DataFrame, 
                            baseline_scores: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for ML model.
        
        Args:
            features_df: Features DataFrame
            baseline_scores: Baseline model scores for synthetic labels
            
        Returns:
            Tuple of (X, y) training arrays
        """
        # Engineer features
        engineered_df = self.feature_engineer.engineer_ml_features(features_df)
        
        # Prepare ML features
        X_df, self.feature_names = self.feature_engineer.prepare_ml_features(
            engineered_df, fit=True
        )
        
        # Create synthetic labels
        y = self._create_synthetic_labels(features_df, baseline_scores)
        
        return X_df.values, y.values
    
    def train(self, features_df: pd.DataFrame, baseline_scores: pd.Series,
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the ML ranking model.
        
        Args:
            features_df: Features DataFrame
            baseline_scores: Baseline scores for synthetic label creation
            validation_split: Fraction of data for validation
            
        Returns:
            dict: Training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare training data
        X, y = self.prepare_training_data(features_df, baseline_scores)
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        if self.model_type == 'lightgbm':
            self.model = lgb.LGBMRanker(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42
            )
            
            # LightGBM requires group information for ranking
            # For simplicity, treat each query as a separate group
            group_train = [len(X_train)]
            group_val = [len(X_val)]
            
            self.model.fit(
                X_train, y_train, group=group_train,
                eval_set=[(X_val, y_val)], eval_group=[group_val],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRanker(
                objective='rank:pairwise',
                learning_rate=0.05,
                max_depth=6,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=42
            )
            
            # XGBoost ranker also needs group information
            group_train = [len(X_train)]
            group_val = [len(X_val)]
            
            self.model.fit(
                X_train, y_train, group=group_train,
                eval_set=[(X_val, y_val)], eval_group=[group_val],
                verbose=False
            )
        
        # Calculate metrics
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'train_corr': np.corrcoef(y_train, y_pred_train)[0, 1],
            'val_corr': np.corrcoef(y_val, y_pred_val)[0, 1]
        }
        
        self.is_trained = True
        logger.info(f"Training completed. Validation RMSE: {metrics['val_rmse']:.4f}")
        
        return metrics
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict scores for agents.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            numpy.ndarray: Predicted scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Engineer features
        engineered_df = self.feature_engineer.engineer_ml_features(features_df)
        
        # Prepare ML features
        X_df, _ = self.feature_engineer.prepare_ml_features(
            engineered_df, fit=False
        )
        
        # Predict
        scores = self.model.predict(X_df.values)
        
        return scores
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Returns:
            pandas.DataFrame: Feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return pd.DataFrame()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_engineer = model_data['feature_engineer']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


class MLRecommender:
    """Complete ML-based recommender system."""
    
    def __init__(self, feature_extractor, ml_ranker: Optional[MLAgentRanker] = None):
        """
        Initialize ML recommender.
        
        Args:
            feature_extractor: FeatureExtractor instance
            ml_ranker: MLAgentRanker instance. If None, creates default LightGBM.
        """
        self.feature_extractor = feature_extractor
        self.ml_ranker = ml_ranker or MLAgentRanker('lightgbm')
    
    def train_model(self, user_queries: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Train the ML model using synthetic data from baseline scores.
        
        Args:
            user_queries: List of synthetic user queries for training.
                         If None, creates default queries.
            
        Returns:
            dict: Training metrics
        """
        if user_queries is None:
            # Create diverse synthetic queries for training
            user_queries = self._generate_synthetic_queries()
        
        all_features = []
        all_baseline_scores = []
        
        # Import baseline scorer
        from .baseline_scorer import BaselineAgentScorer
        baseline_scorer = BaselineAgentScorer()
        
        for query in user_queries:
            features_df = self.feature_extractor.extract_features_for_query(query)
            baseline_scores = baseline_scorer.calculate_baseline_score(features_df)
            
            all_features.append(features_df)
            all_baseline_scores.append(baseline_scores)
        
        # Combine all data
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_scores = pd.concat(all_baseline_scores, ignore_index=True)
        
        # Train model
        metrics = self.ml_ranker.train(combined_features, combined_scores)
        
        return metrics
    
    def _generate_synthetic_queries(self) -> List[Dict[str, Any]]:
        """Generate synthetic user queries for training."""
        # Get unique regions and property types from data
        agents_df = self.feature_extractor.agents_df
        
        all_regions = []
        all_property_types = []
        
        for regions in agents_df['primaryServiceRegions'].dropna():
            if isinstance(regions, list):
                all_regions.extend(regions)
        
        for prop_types in agents_df['propertyTypes'].dropna():
            if isinstance(prop_types, list):
                all_property_types.extend(prop_types)
        
        unique_regions = list(set(all_regions))[:50]  # Limit for efficiency
        unique_property_types = list(set(all_property_types))
        
        # Generate diverse queries
        queries = []
        budgets = [200000, 350000, 500000, 750000, 1000000, 1500000]
        
        for budget in budgets:
            for _ in range(5):  # 5 queries per budget
                num_regions = np.random.randint(1, 4)
                num_prop_types = np.random.randint(1, 3)
                
                selected_regions = np.random.choice(
                    unique_regions, min(num_regions, len(unique_regions)), replace=False
                ).tolist()
                
                selected_prop_types = np.random.choice(
                    unique_property_types, min(num_prop_types, len(unique_property_types)), replace=False
                ).tolist()
                
                query = {
                    'regions': selected_regions,
                    'budget': budget,
                    'property_types': selected_prop_types
                }
                queries.append(query)
        
        return queries
    
    def recommend_agents(self, user_query: Dict[str, Any], top_k: int = 10,
                        explain: bool = False) -> Dict[str, Any]:
        """
        Recommend agents using ML model.
        
        Args:
            user_query: User preferences dictionary
            top_k: Number of agents to recommend
            explain: Whether to include model explanation
            
        Returns:
            dict: Recommendations with optional explanation
        """
        if not self.ml_ranker.is_trained:
            raise ValueError("ML model must be trained before making recommendations")
        
        # Extract features for the query
        features_df = self.feature_extractor.extract_features_for_query(user_query)
        
        # Get ML scores
        ml_scores = self.ml_ranker.predict(features_df)
        
        # Add scores and rank
        result_df = features_df.copy()
        result_df['ml_score'] = ml_scores
        ranked_df = result_df.sort_values('ml_score', ascending=False).head(top_k)
        
        # Prepare recommendations
        recommendations = []
        for _, agent in ranked_df.iterrows():
            rec = {
                'agentId': agent.get('agentId'),
                'name': agent.get('name'),
                'score': agent.get('ml_score'),
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
            'model_type': f'ml_{self.ml_ranker.model_type}'
        }
        
        if explain:
            feature_importance = self.ml_ranker.get_feature_importance()
            result['explanation'] = {
                'model_type': self.ml_ranker.model_type,
                'feature_importance': feature_importance.head(20).to_dict('records') if len(feature_importance) > 0 else [],
                'top_features': feature_importance.head(10)['feature'].tolist() if len(feature_importance) > 0 else []
            }
        
        return result


if __name__ == "__main__":
    print("ML Agent Ranker initialized")
    print("Use with FeatureExtractor from data_preprocessing module")
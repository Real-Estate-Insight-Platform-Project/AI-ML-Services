"""
Skill vector construction and Bayesian quality calibration.

Implements:
- Agent skill vectors combining theme strengths and calibrated sub-scores
- Bayesian shrinkage for rating calibration
- Wilson confidence bounds for review reliability
- Agent Quality Prior (Q_prior) computation
- Robust scaling and normalization
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class SkillVectorBuilder:
    """Builds comprehensive agent skill vectors with quality calibration."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize skill vector builder.
        
        Args:
            confidence_level: Confidence level for Wilson bounds (0.95 = 95%)
        """
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        # Fitted parameters (computed during training)
        self.global_stats = {}
        self.scaling_params = {}
        
    def build_agent_skill_vectors(self,
                                agents_df: pd.DataFrame,
                                agent_theme_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive skill vectors for each agent.
        
        Combines:
        - Calibrated sub-scores (empirical Bayes shrinkage)
        - Theme strengths (sentiment + recency weighted)
        - Volume and stability indicators
        
        Args:
            agents_df: Agent data with aggregated statistics
            agent_theme_matrix: Agent × Theme strength matrix
            
        Returns:
            DataFrame with skill vectors for each agent
        """
        logger.info("Building agent skill vectors...")
        
        # Start with agent data
        skill_df = agents_df.set_index('advertiser_id').copy()
        
        # Add theme strengths
        theme_cols = [col for col in agent_theme_matrix.columns if col.startswith('theme_')]
        for col in theme_cols:
            skill_df[col] = agent_theme_matrix[col].reindex(skill_df.index, fill_value=0.0)
        
        # Prepare sub-score features (use shrunk versions if available)
        sub_score_cols = []
        for base_col in ['avg_responsiveness', 'avg_negotiation', 'avg_professionalism', 'avg_market_expertise']:
            shrunk_col = f"{base_col}_shrunk"
            if shrunk_col in skill_df.columns:
                skill_df[f"skill_{base_col.replace('avg_', '')}"] = skill_df[shrunk_col]
                sub_score_cols.append(f"skill_{base_col.replace('avg_', '')}")
            elif base_col in skill_df.columns:
                # Fill NaNs with global mean for agents without reviews
                global_mean = skill_df[base_col].mean()
                skill_df[f"skill_{base_col.replace('avg_', '')}"] = skill_df[base_col].fillna(global_mean)
                sub_score_cols.append(f"skill_{base_col.replace('avg_', '')}")
        
        # Add stability indicators
        skill_df['volume_score'] = self._calculate_volume_score(skill_df['review_count_actual'])
        skill_df['recency_score_normalized'] = self._normalize_recency_score(skill_df['recency_score'])
        skill_df['experience_score'] = self._calculate_experience_score(skill_df['experience_years'])
        
        # Compute Agent Quality Prior (Q_prior)
        skill_df['q_prior'] = self._calculate_q_prior(skill_df)
        
        # Apply robust scaling to all numeric features
        feature_cols = sub_score_cols + theme_cols + ['volume_score', 'recency_score_normalized', 'experience_score']
        skill_df = self._apply_robust_scaling(skill_df, feature_cols)
        
        logger.info(f"Built skill vectors with {len(feature_cols)} features for {len(skill_df)} agents")
        
        # Store feature names for later use
        self.feature_columns = feature_cols
        self.sub_score_columns = sub_score_cols
        self.theme_columns = theme_cols
        
        return skill_df.reset_index()
    
    def _calculate_volume_score(self, review_counts: pd.Series) -> pd.Series:
        """Calculate volume score with diminishing returns."""
        # Log transform with smoothing to handle zeros
        volume_scores = np.log1p(review_counts)
        
        # Normalize to 0-1 range
        max_score = volume_scores.max()
        if max_score > 0:
            volume_scores = volume_scores / max_score
        
        return volume_scores
    
    def _normalize_recency_score(self, recency_scores: pd.Series) -> pd.Series:
        """Normalize recency scores to 0-1 range."""
        # Fill NaN with 0 (agents with no reviews)
        recency_filled = recency_scores.fillna(0.0)
        
        # Normalize
        max_score = recency_filled.max()
        if max_score > 0:
            return recency_filled / max_score
        else:
            return recency_filled
    
    def _calculate_experience_score(self, experience_years: pd.Series) -> pd.Series:
        """Calculate experience score with diminishing returns."""
        # Fill NaN with median
        exp_filled = experience_years.fillna(experience_years.median())
        
        # Apply square root transform (diminishing returns)
        exp_scores = np.sqrt(np.maximum(0, exp_filled))
        
        # Normalize to 0-1 range
        max_score = exp_scores.max()
        if max_score > 0:
            exp_scores = exp_scores / max_score
        
        return exp_scores
    
    def _calculate_q_prior(self, agents_df: pd.DataFrame) -> pd.Series:
        """
        Calculate Agent Quality Prior combining:
        - Bayesian posterior mean rating
        - Wilson lower bound for confidence
        - Recency factor
        """
        logger.info("Calculating Agent Quality Prior (Q_prior)...")
        
        # Component 1: Bayesian posterior mean rating
        posterior_means = self._bayesian_rating_shrinkage(
            agents_df['avg_rating'].fillna(0.0),
            agents_df['review_count_actual']
        )
        
        # Component 2: Wilson lower bound
        wilson_bounds = agents_df['wilson_lower_bound'].fillna(0.0)
        
        # Component 3: Recency factor
        recency_factors = agents_df['recency_score'].fillna(0.0)
        
        # Normalize components to 0-1 range
        posterior_means_norm = self._min_max_normalize(posterior_means)
        wilson_bounds_norm = self._min_max_normalize(wilson_bounds)
        recency_factors_norm = self._min_max_normalize(recency_factors)
        
        # Weighted combination (these weights could be learned from data)
        q_prior = (0.5 * posterior_means_norm + 
                  0.3 * wilson_bounds_norm + 
                  0.2 * recency_factors_norm)
        
        return q_prior
    
    def _bayesian_rating_shrinkage(self, 
                                 sample_means: pd.Series,
                                 sample_counts: pd.Series,
                                 prior_weight: float = 10.0) -> pd.Series:
        """
        Apply Bayesian shrinkage to agent ratings.
        
        Args:
            sample_means: Agent average ratings
            sample_counts: Number of reviews per agent
            prior_weight: Weight of prior (higher = more shrinkage)
            
        Returns:
            Shrunk ratings (posterior means)
        """
        # Calculate global prior (overall mean rating)
        valid_ratings = sample_means[sample_means > 0]
        if len(valid_ratings) > 0:
            global_mean = valid_ratings.mean()
        else:
            global_mean = 4.0  # Reasonable default
        
        # Bayesian shrinkage: posterior = (n * sample_mean + w * prior) / (n + w)
        posterior_means = (
            (sample_counts * sample_means + prior_weight * global_mean) /
            (sample_counts + prior_weight)
        )
        
        return posterior_means
    
    def _min_max_normalize(self, series: pd.Series) -> pd.Series:
        """Min-max normalize series to 0-1 range."""
        min_val = series.min()
        max_val = series.max()
        
        if max_val > min_val:
            return (series - min_val) / (max_val - min_val)
        else:
            return series * 0  # All zeros if no variation
    
    def _apply_robust_scaling(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Apply robust scaling using median and IQR."""
        logger.info("Applying robust scaling to features...")
        
        scaled_df = df.copy()
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            values = df[col].values
            
            # Calculate robust statistics
            median = np.median(values)
            q75 = np.percentile(values, 75)
            q25 = np.percentile(values, 25)
            iqr = q75 - q25
            
            # Store scaling parameters
            self.scaling_params[col] = {
                'median': median,
                'iqr': iqr if iqr > 0 else 1.0  # Avoid division by zero
            }
            
            # Apply robust scaling: (x - median) / IQR
            scaled_values = (values - median) / self.scaling_params[col]['iqr']
            scaled_df[col] = scaled_values
        
        return scaled_df
    
    def scale_new_data(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Apply previously fitted scaling to new data."""
        scaled_df = df.copy()
        
        for col in feature_cols:
            if col not in df.columns or col not in self.scaling_params:
                continue
            
            params = self.scaling_params[col]
            scaled_df[col] = (df[col] - params['median']) / params['iqr']
        
        return scaled_df
    
    def calculate_confidence_metrics(self, agents_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional confidence and uncertainty metrics."""
        logger.info("Calculating confidence metrics...")
        
        confidence_df = agents_df.copy()
        
        # Wilson score confidence interval width (uncertainty indicator)
        n_reviews = confidence_df['review_count_actual'].fillna(0)
        p_positive = confidence_df['positive_review_ratio'].fillna(0)
        
        # Calculate Wilson interval width
        confidence_df['wilson_ci_width'] = self._wilson_ci_width(p_positive, n_reviews)
        
        # Confidence score (inverse of uncertainty)
        max_width = confidence_df['wilson_ci_width'].max()
        if max_width > 0:
            confidence_df['confidence_score'] = 1 - (confidence_df['wilson_ci_width'] / max_width)
        else:
            confidence_df['confidence_score'] = 1.0
        
        # Data richness score
        confidence_df['data_richness'] = self._calculate_data_richness(confidence_df)
        
        return confidence_df
    
    def _wilson_ci_width(self, p: pd.Series, n: pd.Series) -> pd.Series:
        """Calculate Wilson confidence interval width."""
        # Avoid division by zero
        n_safe = np.maximum(n, 1)
        
        # Wilson interval calculation
        z = self.z_score
        
        # Calculate confidence interval bounds
        denominator = 1 + z**2 / n_safe
        center = (p + z**2 / (2 * n_safe)) / denominator
        
        # Width calculation (simplified)
        width = 2 * z * np.sqrt((p * (1 - p) + z**2 / (4 * n_safe)) / n_safe) / denominator
        
        return width
    
    def _calculate_data_richness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate data richness score based on available information."""
        richness_components = []
        
        # Review volume component
        review_component = np.minimum(df['review_count_actual'] / 20.0, 1.0)  # Cap at 20 reviews
        richness_components.append(review_component * 0.4)
        
        # Recent activity component
        recency_component = df['recency_score'].fillna(0) 
        richness_components.append(recency_component * 0.3)
        
        # Profile completeness component
        profile_score = 0
        for col in ['agent_bio', 'specializations', 'designations', 'languages']:
            if col in df.columns:
                profile_score += (~df[col].isna()).astype(float) * 0.25
        
        if len(richness_components) > 0:
            profile_component = pd.Series(profile_score, index=df.index)
        else:
            profile_component = pd.Series(0.0, index=df.index)
        
        richness_components.append(profile_component * 0.3)
        
        # Combine components
        total_richness = sum(richness_components)
        
        return total_richness
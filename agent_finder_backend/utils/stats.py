import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple
from config.settings import settings


def _clamp_to_range(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Safely clamp a value to a range, handling NaN and inf.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Clamped value within [min_val, max_val]
    """
    if pd.isna(value) or np.isinf(value):
        return min_val
    return min(max_val, max(min_val, float(value)))


def wilson_lower_bound(positive: int, total: int, confidence: float = None) -> float:
    """
    Calculate Wilson score lower bound for positive reviews.
    
    Args:
        positive: Number of positive reviews
        total: Total number of reviews
        confidence: Confidence level (default from settings)
    
    Returns:
        Wilson lower bound score (0-1)
    """
    if confidence is None:
        confidence = settings.WILSON_CONFIDENCE
    
    if total == 0:
        return 0.0
    
    # Ensure positive doesn't exceed total
    positive = min(positive, total)
    
    phat = positive / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    numerator = phat + z * z / (2 * total) - z * np.sqrt(
        (phat * (1 - phat) + z * z / (4 * total)) / total
    )
    denominator = 1 + z * z / total
    
    result = numerator / denominator
    
    # Ensure result is in [0, 1] range and handle edge cases
    return _clamp_to_range(result, 0.0, 1.0)


def bayesian_rating_shrinkage(
    rating: float, 
    count: int, 
    prior_mean: float = None, 
    prior_count: int = None
) -> float:
    """
    Apply Bayesian shrinkage to ratings based on review count.
    
    Agents with few reviews are shrunk toward the prior mean (global average).
    Agents with many reviews retain their rating.
    
    Args:
        rating: Agent's average rating
        count: Number of reviews
        prior_mean: Global average rating (default from settings)
        prior_count: Equivalent sample size for prior (default from settings)
    
    Returns:
        Shrunk rating (clamped to valid range)
    """
    if prior_mean is None:
        prior_mean = settings.BAYESIAN_PRIOR_MEAN
    if prior_count is None:
        prior_count = settings.BAYESIAN_PRIOR_COUNT
    
    # Handle invalid inputs
    if pd.isna(rating) or pd.isna(count):
        return prior_mean
    
    if count == 0:
        return prior_mean
    
    result = (prior_count * prior_mean + count * rating) / (prior_count + count)
    
    # Clamp to rating scale (1-5 stars)
    return _clamp_to_range(result, 1.0, 5.0)


def exponential_decay_score(days: int, decay_rate: float = None) -> float:
    """
    Calculate exponential decay score based on number of days.
    
    Score approaches 0 as days increase.
    
    Args:
        days: Number of days since event
        decay_rate: Daily decay rate (default from settings)
    
    Returns:
        Decay score (0-1)
    """
    if decay_rate is None:
        decay_rate = settings.RECENCY_DECAY_RATE
    
    # Handle invalid inputs
    if pd.isna(days) or days < 0:
        days = 365  # Default to 1 year if invalid
    
    result = np.exp(-decay_rate * days)
    return _clamp_to_range(result, 0.0, 1.0)


def recency_weighted_volume_score(
    count: int, 
    days_since_last: int,
    decay_rate: float = None
) -> float:
    """
    Calculate score combining volume and recency.
    
    Rewards both high volume and recent activity.
    
    Args:
        count: Number of recent transactions
        days_since_last: Days since last transaction
        decay_rate: Daily decay rate
    
    Returns:
        Combined score (0-1)
    """
    if decay_rate is None:
        decay_rate = settings.RECENCY_DECAY_RATE
    
    # Handle invalid inputs
    if pd.isna(count) or count < 0:
        count = 0
    if pd.isna(days_since_last) or days_since_last < 0:
        days_since_last = 365
    
    # Convert to int to avoid issues
    count = int(count)
    days_since_last = int(days_since_last)
    
    # If no sales, return 0
    if count == 0:
        return 0.0
    
    # Normalize count (using log scale to handle wide range)
    # Use 250 as max to accommodate actual data range (some agents have 200+ sales)
    volume_score = min(1.0, np.log1p(count) / np.log1p(250))
    
    # Calculate recency score
    recency_score = exponential_decay_score(days_since_last, decay_rate)
    
    # Combine (geometric mean to penalize either being very low)
    result = np.sqrt(volume_score * recency_score)
    
    # Ensure result is in [0, 1] range
    return _clamp_to_range(result, 0.0, 1.0)


def normalize_score(score: float, min_val: float, max_val: float) -> float:
    """
    Normalize score to 0-1 range.
    
    Args:
        score: Raw score
        min_val: Minimum value in dataset
        max_val: Maximum value in dataset
    
    Returns:
        Normalized score (0-1)
    """
    if max_val == min_val:
        return 1.0
    
    result = (score - min_val) / (max_val - min_val)
    return _clamp_to_range(result, 0.0, 1.0)


def calculate_distance_score(distance_km: float, decay_rate: float = None) -> float:
    """
    Calculate proximity score based on distance.
    
    Closer agents get higher scores.
    
    Args:
        distance_km: Distance in kilometers
        decay_rate: Distance decay rate
    
    Returns:
        Proximity score (0-1)
    """
    if decay_rate is None:
        decay_rate = settings.DISTANCE_DECAY_RATE
    
    # Handle invalid inputs
    if pd.isna(distance_km) or distance_km < 0:
        return 0.0
    
    result = np.exp(-decay_rate * distance_km)
    return _clamp_to_range(result, 0.0, 1.0)


def aggregate_scores(
    scores: np.ndarray,
    weights: np.ndarray,
    method: str = 'weighted_mean'
) -> float:
    """
    Aggregate multiple scores using specified method.
    
    Args:
        scores: Array of scores (0-1)
        weights: Array of weights
        method: Aggregation method ('weighted_mean', 'geometric_mean', 'harmonic_mean')
    
    Returns:
        Aggregated score (0-1)
    """
    # Handle edge cases
    if len(scores) == 0:
        return 0.0
    
    # Remove NaN values
    valid_mask = ~np.isnan(scores) & ~np.isnan(weights)
    scores = scores[valid_mask]
    weights = weights[valid_mask]
    
    if len(scores) == 0:
        return 0.0
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    if method == 'weighted_mean':
        result = np.average(scores, weights=weights)
    elif method == 'geometric_mean':
        # Weighted geometric mean (avoid log(0))
        scores_safe = np.maximum(scores, 1e-10)
        result = np.exp(np.sum(weights * np.log(scores_safe)))
    elif method == 'harmonic_mean':
        # Weighted harmonic mean (avoid division by 0)
        scores_safe = np.maximum(scores, 1e-10)
        result = np.sum(weights) / np.sum(weights / scores_safe)
    else:
        result = np.average(scores, weights=weights)
    
    return _clamp_to_range(result, 0.0, 1.0)
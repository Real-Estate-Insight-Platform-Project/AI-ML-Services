import numpy as np
from scipy import stats
from typing import Tuple
from config.settings import settings


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
    
    phat = positive / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    numerator = phat + z * z / (2 * total) - z * np.sqrt(
        (phat * (1 - phat) + z * z / (4 * total)) / total
    )
    denominator = 1 + z * z / total
    
    return max(0.0, numerator / denominator)


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
        Shrunk rating
    """
    if prior_mean is None:
        prior_mean = settings.BAYESIAN_PRIOR_MEAN
    if prior_count is None:
        prior_count = settings.BAYESIAN_PRIOR_COUNT
    
    if count == 0:
        return prior_mean
    
    return (prior_count * prior_mean + count * rating) / (prior_count + count)


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
    
    return np.exp(-decay_rate * days)


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
        Combined score
    """
    if decay_rate is None:
        decay_rate = settings.RECENCY_DECAY_RATE
    
    # Normalize count (using log scale to handle wide range)
    volume_score = np.log1p(count) / np.log1p(100)  # Max expected ~100
    
    # Calculate recency score
    recency_score = exponential_decay_score(days_since_last, decay_rate)
    
    # Combine (geometric mean to penalize either being very low)
    return np.sqrt(volume_score * recency_score)


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
    return (score - min_val) / (max_val - min_val)


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
    
    return np.exp(-decay_rate * distance_km)


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
        Aggregated score
    """
    # Normalize weights
    weights = weights / np.sum(weights)
    
    if method == 'weighted_mean':
        return np.average(scores, weights=weights)
    elif method == 'geometric_mean':
        # Weighted geometric mean
        return np.exp(np.sum(weights * np.log(scores + 1e-10)))
    elif method == 'harmonic_mean':
        # Weighted harmonic mean
        return np.sum(weights) / np.sum(weights / (scores + 1e-10))
    else:
        return np.average(scores, weights=weights)
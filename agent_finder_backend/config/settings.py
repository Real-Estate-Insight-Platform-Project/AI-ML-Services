from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # Supabase Configuration
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Agent Finder API"
    
    # Model Configuration
    SENTIMENT_MODEL: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Algorithm Parameters
    WILSON_CONFIDENCE: float = 0.95
    BAYESIAN_PRIOR_MEAN: float = 4.0
    BAYESIAN_PRIOR_COUNT: int = 10
    RECENCY_DECAY_RATE: float = 0.01  # Daily decay rate
    REVIEW_RECENCY_DECAY: float = 0.005  # Review age decay
    
    # Distance Configuration
    MAX_DISTANCE_KM: float = 100.0
    DISTANCE_DECAY_RATE: float = 0.02
    
    # Scoring Weights (Base weights, will be adjusted by user preferences)
    WEIGHT_BUYER_SELLER_FIT: float = 0.20
    WEIGHT_PERFORMANCE: float = 0.25
    WEIGHT_REVIEWS: float = 0.20
    WEIGHT_SUB_SCORES: float = 0.15
    WEIGHT_PROXIMITY: float = 0.10
    WEIGHT_ACTIVE_LISTINGS: float = 0.10  # Only for urgent
    
    # Feature Weights within categories
    PERFORMANCE_WEIGHT_RECENCY: float = 0.4
    PERFORMANCE_WEIGHT_VOLUME: float = 0.3
    PERFORMANCE_WEIGHT_EXPERIENCE: float = 0.3
    
    # Filtering
    MIN_REVIEW_COUNT: int = 0
    MAX_RECOMMENDATIONS: int = 20
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
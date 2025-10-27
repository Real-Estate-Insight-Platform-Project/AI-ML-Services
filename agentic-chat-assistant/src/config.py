"""
Configuration Management for Agentic Real Estate Assistant
Handles all environment variables and settings with validation
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    
    # Gemini Models
    gemini_flash_model: str = Field(
        default="gemini-2.5-flash",
        alias="GEMINI_FLASH_MODEL"
    )
    
    # Supabase (PostgreSQL)
    supabase_url: str = Field(..., alias="SQL_AGENT_DATABASE_URL")
    
    # BigQuery
    bq_project_id: str = Field(..., alias="BQ_PROJECT_ID")
    bq_dataset_id: str = Field(
        default="real_estate_market",
        alias="BQ_DATASET_ID"
    )
    bq_location: str = Field(default="US", alias="BQ_LOCATION")
    google_application_credentials: str = Field(
        default="service_keys.json",
        alias="GOOGLE_APPLICATION_CREDENTIALS"
    )
    
    # Redis
    redis_url: str = Field(..., alias="REDIS_URL")
    redis_session_ttl: int = Field(default=3600, alias="REDIS_SESSION_TTL")
    redis_cache_ttl: int = Field(default=300, alias="REDIS_CACHE_TTL")
    
    # Application
    app_name: str = Field(default="Agentic Real Estate Assistant", alias="APP_NAME")
    app_version: str = Field(default="3.0.0", alias="APP_VERSION")
    app_port: int = Field(default=8005, alias="APP_PORT")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")
    
    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8005"],
        alias="CORS_ORIGINS"
    )
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=False, alias="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, alias="RATE_LIMIT_WINDOW")
    
    # Website
    website_base_url: str = Field(
        default="https://yourdomain.com",
        alias="WEBSITE_BASE_URL"
    )
    agent_finder_url: str = Field(
        default="https://real-estate-insights/agent-finder",
        alias="AGENT_FINDER_URL"
    )
    
    # Geocoding
    geocoding_user_agent: str = Field(
        default="RealEstateInsightPlatform/3.0",
        alias="GEOCODING_USER_AGENT"
    )
    
    # Database Connection Pool
    db_pool_size: int = Field(default=10, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, alias="DB_MAX_OVERFLOW")
    db_pool_recycle: int = Field(default=3600, alias="DB_POOL_RECYCLE")
    
    # Query Limits
    max_query_results: int = Field(default=100, alias="MAX_QUERY_RESULTS")
    max_distance_miles: float = Field(default=50.0, alias="MAX_DISTANCE_MILES")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")
    
    @field_validator("google_application_credentials")
    @classmethod
    def set_google_credentials_env(cls, v: str) -> str:
        """Set GOOGLE_APPLICATION_CREDENTIALS environment variable"""
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = v
        return v
    
    @property
    def bigquery_dataset_full(self) -> str:
        """Get fully qualified BigQuery dataset name"""
        return f"{self.bq_project_id}.{self.bq_dataset_id}"


# Global settings instance
settings = Settings()


# Table whitelists for security
ALLOWED_SUPABASE_TABLES = [
    "properties",
    "nri_counties",
    "uszips",
    "gis.us_counties",
]

ALLOWED_BIGQUERY_TABLES = [
    "county_lookup",
    "county_market",
    "county_predictions",
    "state_lookup",
    "state_market",
    "state_predictions",
]

# Protected tables that should never be exposed
PROTECTED_TABLES = [
    "profiles",
    "user_favorites",
    "users",
    "sessions",
    "auth",
]

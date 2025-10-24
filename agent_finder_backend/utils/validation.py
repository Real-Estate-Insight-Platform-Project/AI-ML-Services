"""
Utility functions for validating data before database insertion.
Ensures all values meet PostgreSQL check constraints.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional


# Define constraint ranges for each column
CONSTRAINT_RANGES = {
    # Review metrics (0-1 range)
    'wilson_score': (0.0, 1.0),
    'performance_score': (0.0, 1.0),
    'experience_score': (0.0, 1.0),
    'buyer_satisfaction': (0.0, 1.0),
    'seller_satisfaction': (0.0, 1.0),
    'recency_weight': (0.0, 1.0),
    'sentiment_confidence': (0.0, 1.0),
    
    # Sub-scores (1-5 range, nullable)
    'avg_sub_responsiveness': (1.0, 5.0),
    'avg_sub_negotiation': (1.0, 5.0),
    'avg_sub_professionalism': (1.0, 5.0),
    'avg_sub_market_expertise': (1.0, 5.0),
    
    # Skill scores (0-1 range)
    'skill_communication': (0.0, 1.0),
    'skill_local_knowledge': (0.0, 1.0),
    'skill_attention_to_detail': (0.0, 1.0),
    'skill_patience': (0.0, 1.0),
    'skill_honesty': (0.0, 1.0),
    'skill_problem_solving': (0.0, 1.0),
    'skill_dedication': (0.0, 1.0),
    
    # Negative qualities (0-1 range)
    'negative_unresponsive': (0.0, 1.0),
    'negative_pushy': (0.0, 1.0),
    'negative_unprofessional': (0.0, 1.0),
    'negative_inexperienced': (0.0, 1.0),
    'negative_dishonest': (0.0, 1.0),
    
    # Rating (1-5 range)
    'shrunk_rating': (1.0, 5.0),
    'agent_rating': (1.0, 5.0),
}


def is_valid_value(value: Any) -> bool:
    """
    Check if a value is valid (not NaN, not inf, not None).
    
    Args:
        value: Value to check
    
    Returns:
        True if value is valid
    """
    if value is None:
        return False
    
    try:
        if pd.isna(value):
            return False
        if isinstance(value, (float, np.floating)):
            if np.isinf(value):
                return False
    except (TypeError, ValueError):
        pass
    
    return True


def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to a specified range.
    
    For PostgreSQL NUMERIC(5,4), the maximum is actually 9.9999, not 10.0
    For columns with CHECK(x <= 1), we use 0.9999 as the max to be safe.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Clamped value
    """
    if not is_valid_value(value):
        return min_val
    
    # For values that should be <= 1.0, use 0.9999 to avoid precision issues
    if max_val == 1.0:
        max_val = 0.9999
    
    # For values that should be <= 5.0, use 4.99 to be safe
    if max_val == 5.0:
        max_val = 4.99
    
    return float(min(max_val, max(min_val, value)))


def validate_and_clamp(column_name: str, value: Any) -> Optional[float]:
    """
    Validate and clamp a value according to its database constraints.
    
    Args:
        column_name: Name of the database column
        value: Value to validate
    
    Returns:
        Validated and clamped value, or None if column has no constraints
    """
    if column_name not in CONSTRAINT_RANGES:
        # No specific constraint, just check if valid
        if not is_valid_value(value):
            return None
        return value
    
    min_val, max_val = CONSTRAINT_RANGES[column_name]
    
    if not is_valid_value(value):
        # Return minimum value for invalid inputs
        return clamp_value(min_val, min_val, max_val)
    
    return clamp_value(value, min_val, max_val)


def validate_row_for_db(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate all values in a row dictionary before database insertion.
    
    This ensures:
    - No NaN or inf values
    - All constrained columns are within valid ranges
    - Integer columns are properly converted
    - Array columns are properly formatted
    
    Args:
        row_dict: Dictionary of column: value pairs
    
    Returns:
        Validated dictionary ready for database insertion
    """
    validated = {}
    
    for key, value in row_dict.items():
        # Skip None values
        if value is None:
            continue
        
        # Handle integer columns
        if key in ['positive_review_count', 'negative_review_count', 'neutral_review_count',
                   'buyer_review_count', 'seller_review_count', 'buyer_positive_count', 
                   'seller_positive_count', 'sub_responsiveness_count', 'sub_negotiation_count',
                   'sub_professionalism_count', 'sub_market_expertise_count', 'days_since_review',
                   'days_since_last_sale']:
            if is_valid_value(value):
                try:
                    validated[key] = int(value)
                except (ValueError, TypeError):
                    validated[key] = 0
            else:
                validated[key] = 0
            continue
        
        # Handle bigint columns (prices)
        if key in ['min_price', 'max_price', 'active_listings_min_price', 
                   'active_listings_max_price', 'recently_sold_min_price', 
                   'recently_sold_max_price']:
            if is_valid_value(value):
                try:
                    validated[key] = int(value)
                except (ValueError, TypeError):
                    pass  # Skip invalid price values
            continue
        
        # Handle constrained float columns
        if key in CONSTRAINT_RANGES:
            validated_val = validate_and_clamp(key, value)
            if validated_val is not None:
                validated[key] = validated_val
            continue
        
        # Handle array columns
        if key in ['property_types', 'additional_specializations', 'service_zipcodes_list']:
            if value is not None and hasattr(value, '__len__'):
                try:
                    # Convert to list if not already
                    if isinstance(value, list):
                        validated[key] = value
                    else:
                        validated[key] = list(value)
                except (TypeError, ValueError):
                    pass
            continue
        
        # Handle string columns that must be in specific set
        if key == 'sentiment':
            if value in ['positive', 'negative', 'neutral']:
                validated[key] = value
            continue
        
        # Handle other columns - pass through if valid
        if is_valid_value(value):
            if isinstance(value, (int, np.integer)):
                validated[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                validated[key] = float(value)
            else:
                validated[key] = value
    
    return validated


def validate_dataframe_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    """
    Validate and fix all values in a DataFrame column.
    
    Args:
        df: DataFrame containing the column
        column_name: Name of the column to validate
    
    Returns:
        Series with validated values
    """
    if column_name not in df.columns:
        return pd.Series()
    
    if column_name in CONSTRAINT_RANGES:
        min_val, max_val = CONSTRAINT_RANGES[column_name]
        
        # Replace invalid values
        validated = df[column_name].copy()
        
        # Replace NaN with minimum value
        validated = validated.fillna(min_val)
        
        # Replace inf with maximum value  
        validated = validated.replace([np.inf], max_val)
        validated = validated.replace([-np.inf], min_val)
        
        # Clamp to range (accounting for NUMERIC precision)
        safe_max = max_val if max_val != 1.0 else 0.9999
        safe_max = safe_max if safe_max != 5.0 else 4.99
        validated = validated.clip(lower=min_val, upper=safe_max)
        
        return validated
    
    return df[column_name]
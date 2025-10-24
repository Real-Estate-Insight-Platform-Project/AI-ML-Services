from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from enum import Enum


class UserType(str, Enum):
    """User type enum."""
    BUYER = "buyer"
    SELLER = "seller"


class PropertyType(str, Enum):
    """Property type enum."""
    SINGLE_FAMILY = "single_family"
    MULTI_FAMILY = "multi_family"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    LAND = "land"
    COMMERCIAL = "commercial"
    LUXURY = "luxury"
    NEW_CONSTRUCTION = "new_construction"


class SubScore(str, Enum):
    """Sub-score categories."""
    RESPONSIVENESS = "responsiveness"
    NEGOTIATION = "negotiation"
    PROFESSIONALISM = "professionalism"
    MARKET_EXPERTISE = "market_expertise"


class Skill(str, Enum):
    """Additional skill categories."""
    COMMUNICATION = "communication"
    LOCAL_KNOWLEDGE = "local_knowledge"
    ATTENTION_TO_DETAIL = "attention_to_detail"
    PATIENCE = "patience"
    HONESTY = "honesty"
    PROBLEM_SOLVING = "problem_solving"
    DEDICATION = "dedication"


class AdditionalSpecialization(str, Enum):
    """Additional specialization categories."""
    FIRST_TIME_BUYER = "first_time_buyer"
    INVESTOR = "investor"
    VETERAN = "veteran"
    SENIOR = "senior"
    RELOCATION = "relocation"
    FORECLOSURE = "foreclosure"
    SHORT_SALE = "short_sale"
    RENTAL = "rental"


class PreferenceWeight(BaseModel):
    """Weight for a preference."""
    name: str
    weight: float = Field(ge=0.0, le=1.0, description="Weight between 0 and 1")


class AgentSearchRequest(BaseModel):
    """Request model for agent search."""
    
    # Required fields
    user_type: UserType = Field(..., description="Whether user is buying or selling")
    state: str = Field(..., description="State (2-letter code or full name)")
    city: str = Field(..., description="City name")
    
    # Price range
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price")
    
    # Property details
    property_type: Optional[PropertyType] = Field(None, description="Type of property")
    
    # Urgency
    is_urgent: bool = Field(False, description="Whether need is urgent")
    
    # Language
    language: Optional[str] = Field("English", description="Preferred language")
    
    # Preferences (user-selected weights)
    sub_score_preferences: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="User weights for sub-scores (responsiveness, negotiation, etc.)"
    )
    
    skill_preferences: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="User weights for skills (communication, local_knowledge, etc.)"
    )
    
    # Additional specializations (optional filters)
    additional_specializations: Optional[List[str]] = Field(
        default_factory=list,
        description="Additional specializations to prefer"
    )
    
    # Pagination
    max_results: int = Field(20, ge=1, le=50, description="Maximum number of results")
    
    @validator('max_price')
    def validate_price_range(cls, v, values):
        """Validate that max_price > min_price."""
        if v is not None and 'min_price' in values and values['min_price'] is not None:
            if v < values['min_price']:
                raise ValueError("max_price must be greater than min_price")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "user_type": "buyer",
                "state": "CA",
                "city": "Los Angeles",
                "min_price": 500000,
                "max_price": 1000000,
                "property_type": "single_family",
                "is_urgent": True,
                "language": "English",
                "sub_score_preferences": {
                    "responsiveness": 0.4,
                    "negotiation": 0.3,
                    "professionalism": 0.2,
                    "market_expertise": 0.1
                },
                "skill_preferences": {
                    "communication": 0.5,
                    "local_knowledge": 0.5
                },
                "additional_specializations": ["first_time_buyer"],
                "max_results": 10
            }
        }


class AgentRecommendation(BaseModel):
    """Single agent recommendation."""
    
    advertiser_id: int
    full_name: str
    state: str
    agent_base_city: str
    agent_base_zipcode: Optional[str]
    
    # Contact info
    phone_primary: Optional[str]
    office_phone: Optional[str]
    agent_website: Optional[str]
    office_name: Optional[str]
    
    # Profile
    has_photo: bool
    agent_photo_url: Optional[str]
    experience_years: Optional[float]
    
    # Scores
    matching_score: float = Field(..., description="Overall matching score (0-100)")
    proximity_score: float = Field(..., description="Proximity score (0-1)")
    distance_km: Optional[float] = Field(None, description="Distance in kilometers")
    
    # Statistics
    review_count: int
    agent_rating: float
    positive_review_count: int
    negative_review_count: int
    
    # Performance
    recently_sold_count: int
    active_listings_count: int
    days_since_last_sale: Optional[int]
    
    # Specializations
    property_types: List[str] = Field(default_factory=list)
    additional_specializations: List[str] = Field(default_factory=list)
    
    # Sub-scores
    avg_responsiveness: Optional[float]
    avg_negotiation: Optional[float]
    avg_professionalism: Optional[float]
    avg_market_expertise: Optional[float]
    
    # Buyer/seller fit
    buyer_seller_fit: str = Field(..., description="'buyer', 'seller', or 'both'")
    
    class Config:
        schema_extra = {
            "example": {
                "advertiser_id": 3322174,
                "full_name": "Michelle Crew",
                "state": "AK",
                "agent_base_city": "Wasilla",
                "agent_base_zipcode": "99654",
                "phone_primary": "(907) 521-6474",
                "office_phone": "(907) 376-2414",
                "agent_website": "https://www.michellecrew.com",
                "office_name": "Jack White Real Estate Mat Su",
                "has_photo": True,
                "agent_photo_url": "https://example.com/photo.jpg",
                "experience_years": 7.0,
                "matching_score": 87.5,
                "proximity_score": 0.92,
                "distance_km": 5.3,
                "review_count": 20,
                "agent_rating": 4.8,
                "positive_review_count": 18,
                "negative_review_count": 1,
                "recently_sold_count": 20,
                "active_listings_count": 3,
                "days_since_last_sale": 18,
                "property_types": ["single_family", "land"],
                "additional_specializations": ["first_time_buyer"],
                "avg_responsiveness": 4.9,
                "avg_negotiation": 4.8,
                "avg_professionalism": 5.0,
                "avg_market_expertise": 4.7,
                "buyer_seller_fit": "both"
            }
        }


class AgentSearchResponse(BaseModel):
    """Response model for agent search."""
    
    success: bool
    message: str
    total_results: int
    recommendations: List[AgentRecommendation]
    
    # Search metadata
    search_params: Dict
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Found 10 matching agents",
                "total_results": 10,
                "recommendations": [],
                "search_params": {
                    "user_type": "buyer",
                    "state": "CA",
                    "city": "Los Angeles"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    success: bool = False
    error: str
    details: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid state code",
                "details": "State 'XX' not found in database"
            }
        }
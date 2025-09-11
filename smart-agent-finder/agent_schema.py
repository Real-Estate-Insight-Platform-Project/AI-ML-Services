from typing import List, Optional
from pydantic import BaseModel, Field

class Review(BaseModel):
    text: Optional[str] = ""

class Agent(BaseModel):
    name: Optional[str] = ""
    work_title: Optional[str] = ""           # the “second line” under name
    years_experience: Optional[str] = ""     # free text; normalize later
    recent_sales_12mo: Optional[str] = ""
    price_range: Optional[str] = ""
    overall_rating: Optional[float] = None
    review_count: Optional[int] = None
    reviews: List[Review] = Field(default_factory=list)
    active_listing_urls: List[str] = Field(default_factory=list)
    phones: List[str] = Field(default_factory=list)
    location: Optional[str] = ""
    profile_url: Optional[str] = ""
    state: Optional[str] = ""
    zip: Optional[str] = ""

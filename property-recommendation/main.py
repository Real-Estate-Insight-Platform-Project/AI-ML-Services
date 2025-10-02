from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

from supabase_client import SupabaseClient
from recommendation_engine import PropertyRecommendationEngine

app = FastAPI(title="Property Recommendation API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://localhost:3001"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
supabase_client = SupabaseClient()
recommendation_engine = PropertyRecommendationEngine()


class RecommendationRequest(BaseModel):
    property_id: str
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 6


class RecommendationResponse(BaseModel):
    target_property: Dict[str, Any]
    similar_properties: List[Dict[str, Any]]
    total_recommendations: int


@app.get("/")
async def root():
    return {"message": "Property Recommendation API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        # Get target property
        target_property = supabase_client.get_property_by_id(
            request.property_id)
        if not target_property:
            raise HTTPException(status_code=404, detail="Property not found")

        # Get all active properties (you might want to cache this)
        all_properties = supabase_client.get_properties()

        # Get similar properties
        similar_properties = recommendation_engine.recommend_similar_properties(
            target_property=target_property,
            all_properties=all_properties,
            filters=request.filters,
            limit=request.limit
        )

        return RecommendationResponse(
            target_property=target_property,
            similar_properties=similar_properties,
            total_recommendations=len(similar_properties)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/properties")
async def get_properties(
    city: Optional[str] = None,
    property_type: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    min_bedrooms: Optional[int] = None,
    min_bathrooms: Optional[float] = None
):
    filters = {
        "city": city,
        "property_type": property_type,
        "min_price": min_price,
        "max_price": max_price,
        "min_bedrooms": min_bedrooms,
        "min_bathrooms": min_bathrooms
    }

    properties = supabase_client.get_properties(filters)
    return {"properties": properties}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

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

# Initialize clients with error handling
try:
    supabase_client = SupabaseClient()
    recommendation_engine = PropertyRecommendationEngine()
    print("✅ Supabase client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing Supabase client: {e}")
    supabase_client = None
    recommendation_engine = None


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
    return {"message": "Property Recommendation API", "status": "running"}


@app.get("/health")
async def health_check():
    if supabase_client is None:
        raise HTTPException(
            status_code=500, detail="Supabase client not initialized")

    # Test connection by fetching one property
    try:
        test_properties = supabase_client.get_properties(limit=1)
        return {
            "status": "healthy",
            "supabase_connected": True,
            "test_properties_count": len(test_properties)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Supabase connection failed: {str(e)}")


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    if supabase_client is None or recommendation_engine is None:
        raise HTTPException(
            status_code=500, detail="Service not properly initialized")

    try:
        # Get target property
        target_property = supabase_client.get_property_by_id(
            request.property_id)
        if not target_property:
            raise HTTPException(status_code=404, detail="Property not found")

        # Get all active properties
        all_properties = supabase_client.get_properties()

        if not all_properties:
            raise HTTPException(
                status_code=404, detail="No properties found in database")

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

    except HTTPException:
        raise
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
    if supabase_client is None:
        raise HTTPException(
            status_code=500, detail="Service not properly initialized")

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
    print("🚀 Starting Property Recommendation API...")
    print("📝 Make sure you have set SUPABASE_URL and SUPABASE_KEY in your .env file")
    uvicorn.run(app, host="0.0.0.0", port=8000)

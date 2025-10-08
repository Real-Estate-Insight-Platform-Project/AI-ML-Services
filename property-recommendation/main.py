from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

# Load environment variables from .env file
load_dotenv()

# Database connection
DATABASE_URL = os.getenv("POSTGRES_URL")
if not DATABASE_URL:
    raise RuntimeError("POSTGRES_URL is not set in the environment variables.")

engine = create_engine(DATABASE_URL)

app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:3000",  # Your Next.js frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class Property(BaseModel):
    id: str
    title: str
    price: float
    bedrooms: int
    bathrooms: float
    square_feet: int
    latitude_coordinates: Optional[float] = None
    longitude_coordinates: Optional[float] = None
    property_image: Optional[str] = None


class RecommendationRequest(BaseModel):
    property_id: str
    filters: Optional[dict] = None
    limit: int = 6


class RecommendationResponse(BaseModel):
    target_property: Property
    similar_properties: List[Property]


def get_property_by_id(property_id: str) -> Optional[Property]:
    with engine.connect() as connection:
        query = text("SELECT id, title, price, bedrooms, bathrooms, square_feet, latitude_coordinates, longitude_coordinates, property_image FROM properties WHERE id = :property_id")
        result = connection.execute(
            query, {"property_id": property_id}).fetchone()
        if result:
            # Manually map RowProxy to a dictionary before creating the Pydantic model
            result_dict = {
                "id": str(result.id),
                "title": result.title,
                "price": result.price,
                "bedrooms": result.bedrooms,
                "bathrooms": result.bathrooms,
                "square_feet": result.square_feet,
                "latitude_coordinates": result.latitude_coordinates,
                "longitude_coordinates": result.longitude_coordinates,
                "property_image": result.property_image,
            }
            return Property(**result_dict)
        return None


@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    target_property = get_property_by_id(request.property_id)
    if not target_property:
        raise HTTPException(status_code=404, detail="Property not found")

    with engine.connect() as connection:
        # Start building the query
        query_str = """
            SELECT
                id,
                title,
                price,
                bedrooms,
                bathrooms,
                square_feet,
                latitude_coordinates,
                longitude_coordinates,
                property_image,
                (
                    6371 * acos(
                        cos(radians(:lat)) * cos(radians(latitude_coordinates)) *
                        cos(radians(longitude_coordinates) - radians(:lon)) +
                        sin(radians(:lat)) * sin(radians(latitude_coordinates))
                    )
                ) AS distance
            FROM
                properties
            WHERE
                id != :target_id
        """
        params = {
            "lat": target_property.latitude_coordinates,
            "lon": target_property.longitude_coordinates,
            "target_id": target_property.id,
            "price": target_property.price,
            "bedrooms": target_property.bedrooms,
            "bathrooms": target_property.bathrooms,
        }

        # Add filters to the query
        if request.filters:
            if request.filters.get("city"):
                query_str += " AND city = :city"
                params["city"] = request.filters["city"]
            if request.filters.get("property_type"):
                query_str += " AND property_type = :property_type"
                params["property_type"] = request.filters["property_type"]
            if request.filters.get("min_price"):
                query_str += " AND price >= :min_price"
                params["min_price"] = float(request.filters["min_price"])
            if request.filters.get("max_price"):
                query_str += " AND price <= :max_price"
                params["max_price"] = float(request.filters["max_price"])
            if request.filters.get("min_bedrooms"):
                query_str += " AND bedrooms >= :min_bedrooms"
                params["min_bedrooms"] = int(request.filters["min_bedrooms"])
            if request.filters.get("min_bathrooms"):
                query_str += " AND bathrooms >= :min_bathrooms"
                params["min_bathrooms"] = float(
                    request.filters["min_bathrooms"])

        # Add ordering and limit
        query_str += " ORDER BY distance ASC, abs(price - :price) ASC, abs(bedrooms - :bedrooms) ASC, abs(bathrooms - :bathrooms) ASC LIMIT :limit"
        params["limit"] = request.limit

        query = text(query_str)
        similar_properties_result = connection.execute(
            query, params).fetchall()

        similar_properties = [
            Property(**{
                "id": str(row.id),
                "title": row.title,
                "price": row.price,
                "bedrooms": row.bedrooms,
                "bathrooms": row.bathrooms,
                "square_feet": row.square_feet,
                "latitude_coordinates": row.latitude_coordinates,
                "longitude_coordinates": row.longitude_coordinates,
                "property_image": row.property_image,
            }) for row in similar_properties_result
        ]

    return RecommendationResponse(
        target_property=target_property,
        similar_properties=similar_properties,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

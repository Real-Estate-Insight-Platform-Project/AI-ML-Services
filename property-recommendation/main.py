import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Environment and Database Setup ---
load_dotenv()
DATABASE_URL = os.getenv("POSTGRES_URL")
if not DATABASE_URL:
    raise RuntimeError("POSTGRES_URL is not set in the environment variables.")
engine = create_engine(DATABASE_URL)

app = FastAPI()

# --- CORS Configuration ---
origins = ["http://34.72.69.249:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---


class Property(BaseModel):
    id: str
    title: str
    price: float
    bedrooms: int
    bathrooms: float
    square_feet: int
    property_image: Optional[str] = None
    latitude_coordinates: Optional[float] = None
    longitude_coordinates: Optional[float] = None
    similarity_score: Optional[float] = None


class RecommendationRequest(BaseModel):
    property_id: str
    filters: Optional[dict] = None
    limit: int = 6


class RecommendationResponse(BaseModel):
    target_property: Property
    similar_properties: List[Property]


# --- In-Memory Cache for ML Model ---
ml_data = {
    "properties_df": pd.DataFrame(),
    "similarity_matrix": None,
}

# --- Machine Learning Pipeline ---


def build_feature_pipeline():
    """Builds the scikit-learn pipeline for feature transformation."""
    numerical_features = ['price', 'bedrooms', 'bathrooms', 'square_feet',
                          'year_built', 'latitude_coordinates', 'longitude_coordinates']
    categorical_features = ['property_type', 'city']
    textual_features = 'text_features'  # Combined title and description

    # Create transformers
    numerical_transformer = MinMaxScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    textual_transformer = TfidfVectorizer(
        stop_words='english', max_features=500)

    # Create a preprocessor with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', textual_transformer, textual_features)
        ],
        remainder='drop'
    )
    return preprocessor


@app.on_event("startup")
def load_and_preprocess_data():
    """Load data from the database and create the similarity matrix on startup."""
    print("Loading properties and building recommendation model...")
    with engine.connect() as connection:
        query = text("SELECT id, title, description, price, bedrooms, bathrooms, square_feet, year_built, property_type, city, state, latitude_coordinates, longitude_coordinates, property_image, property_hyperlink FROM properties WHERE listing_status = 'active'")
        properties_df = pd.read_sql(query, connection)

    # --- Data Cleaning and Feature Engineering ---
    properties_df['id'] = properties_df['id'].astype(str)

    # Fill missing numerical values with the median of their respective columns
    for col in ['price', 'bedrooms', 'bathrooms', 'square_feet', 'year_built', 'latitude_coordinates', 'longitude_coordinates']:
        if properties_df[col].isnull().any():
            median_val = properties_df[col].median()
            properties_df[col].fillna(median_val, inplace=True)

    # Combine text features
    properties_df['description'] = properties_df['description'].fillna('')
    properties_df['title'] = properties_df['title'].fillna('')
    properties_df['text_features'] = properties_df['title'] + \
        ' ' + properties_df['description']

    # --- Build and Apply ML Pipeline ---
    pipeline = build_feature_pipeline()
    feature_matrix = pipeline.fit_transform(properties_df)

    # --- Calculate Cosine Similarity ---
    similarity_matrix = cosine_similarity(feature_matrix)

    # --- Store in "cache" ---
    ml_data["properties_df"] = properties_df
    ml_data["similarity_matrix"] = similarity_matrix
    print("Recommendation model built successfully.")


# --- API Endpoints ---
def get_property_by_id_from_df(property_id: str) -> Optional[pd.Series]:
    """Retrieves a single property from the cached DataFrame."""
    property_series = ml_data["properties_df"][ml_data["properties_df"]
                                               ["id"] == property_id]
    if not property_series.empty:
        return property_series.iloc[0]
    return None


@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    properties_df = ml_data["properties_df"]
    similarity_matrix = ml_data["similarity_matrix"]

    if properties_df.empty or similarity_matrix is None:
        raise HTTPException(
            status_code=503, detail="Recommendation model is not ready.")

    target_property_series = get_property_by_id_from_df(request.property_id)
    if target_property_series is None:
        raise HTTPException(status_code=404, detail="Property not found")

    # Convert pandas Series to Pydantic model
    target_property = Property(**target_property_series.to_dict())

    # Find the index of the target property in the DataFrame
    try:
        target_idx = properties_df.index[properties_df['id'] == request.property_id].tolist()[
            0]
    except IndexError:
        raise HTTPException(
            status_code=404, detail="Property not found in the model.")

    # Get similarity scores for the target property
    sim_scores = list(enumerate(similarity_matrix[target_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar properties
    similar_indices = [i[0] for i in sim_scores if i[0] != target_idx]

    # Filter results based on request filters
    filtered_df = properties_df.iloc[similar_indices].copy()
    if request.filters:
        filters = request.filters
        if filters.get("city"):
            filtered_df = filtered_df[filtered_df['city'] == filters["city"]]
        if filters.get("property_type"):
            filtered_df = filtered_df[filtered_df['property_type']
                                      == filters["property_type"]]
        if filters.get("min_price"):
            filtered_df = filtered_df[filtered_df['price'] >= float(
                filters["min_price"])]
        if filters.get("max_price"):
            filtered_df = filtered_df[filtered_df['price'] <= float(
                filters["max_price"])]
        if filters.get("min_bedrooms"):
            filtered_df = filtered_df[filtered_df['bedrooms'] >= int(
                filters["min_bedrooms"])]
        if filters.get("min_bathrooms"):
            filtered_df = filtered_df[filtered_df['bathrooms'] >= float(
                filters["min_bathrooms"])]

    # Get top N results after filtering
    top_results_df = filtered_df.head(request.limit)

    # Add similarity scores to the results
    result_indices = top_results_df.index.tolist()
    final_sim_scores = {idx: score for idx, score in sim_scores}
    top_results_df['similarity_score'] = [final_sim_scores[i]
                                          for i in result_indices]

    similar_properties = [Property(**row)
                          for index, row in top_results_df.iterrows()]

    return RecommendationResponse(
        target_property=target_property,
        similar_properties=similar_properties,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)

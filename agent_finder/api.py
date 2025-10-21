"""
FastAPI service for the Agent Finder system.

Provides REST API endpoints for:
- Agent recommendations with preferences and filters
- System status and statistics
- Model training and management
- Health checks and monitoring
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import os
import pandas as pd
from datetime import datetime
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.agent_finder import AgentFinder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agent Finder API",
    description="Intelligent agent recommendation system with ML-powered matching",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent finder instance
agent_finder: Optional[AgentFinder] = None

# Global sentiment analyzer
sentiment_analyzer = None

# Pydantic models for API

class UserPreferences(BaseModel):
    """User preference sliders for agent matching."""
    responsiveness: float = Field(0.5, ge=0.0, le=1.0, description="Importance of agent responsiveness (0-1)")
    negotiation: float = Field(0.5, ge=0.0, le=1.0, description="Importance of negotiation skills (0-1)")
    professionalism: float = Field(0.5, ge=0.0, le=1.0, description="Importance of professionalism (0-1)")
    market_expertise: float = Field(0.5, ge=0.0, le=1.0, description="Importance of market expertise (0-1)")

class UserFilters(BaseModel):
    """User filters and requirements for agent search."""
    state: Optional[str] = Field(None, description="Required state (e.g., 'AK', 'CA')")
    city: Optional[str] = Field(None, description="Required or preferred city")
    transaction_type: Optional[str] = Field(None, description="'buying' or 'selling'")
    price_min: Optional[float] = Field(None, ge=0, description="Minimum price range")
    price_max: Optional[float] = Field(None, ge=0, description="Maximum price range")
    language: Optional[str] = Field("English", description="Required language")
    specialization: Optional[str] = Field(None, description="Preferred specialization")
    min_rating: Optional[float] = Field(None, ge=1.0, le=5.0, description="Minimum agent rating")
    min_reviews: Optional[int] = Field(None, ge=0, description="Minimum number of reviews")
    require_recent_activity: bool = Field(False, description="Require recent client activity")
    active_only: bool = Field(True, description="Only active agents")

class RecommendationRequest(BaseModel):
    """Request for agent recommendations."""
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    filters: UserFilters = Field(default_factory=UserFilters)
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations")
    include_explanations: bool = Field(True, description="Include detailed explanations")

class AgentProfile(BaseModel):
    """Agent profile information."""
    state: str
    city: str
    rating: float
    review_count: int
    experience_years: float
    specializations: str
    languages: str
    agent_type: str
    office_name: str
    phone: str
    website: str
    bio: str

class AgentMetrics(BaseModel):
    """Agent performance metrics."""
    responsiveness: float
    negotiation: float
    professionalism: float
    market_expertise: float
    q_prior: float
    wilson_lower_bound: float
    recency_score: float

class AgentRecommendation(BaseModel):
    """Single agent recommendation."""
    agent_id: int
    name: str
    rank: int
    utility_score: float
    availability_fit: float
    confidence_score: float
    profile: AgentProfile
    metrics: AgentMetrics

class RecommendationResponse(BaseModel):
    """Complete recommendation response."""
    recommendations: List[AgentRecommendation]
    explanations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    summary: Dict[str, Any]

class TrainingRequest(BaseModel):
    """Request to train the system."""
    use_cache: bool = Field(True, description="Use cached intermediate results")
    save_cache: bool = Field(True, description="Save results to cache")

class ReviewSentiment(BaseModel):
    """Classified review with sentiment."""
    review_id: str
    review_text: str
    review_rating: float
    sentiment: str  # 'good', 'bad', 'neutral'
    sentiment_score: float  # confidence score
    review_date: str
    reviewer_role: str
    sub_scores: Dict[str, Optional[float]]

class ReviewSentimentResponse(BaseModel):
    """Response for agent review sentiment classification."""
    agent_id: int
    agent_name: str
    total_reviews: int  # Total reviews for this agent
    recent_reviews_count: int  # Number of recent reviews returned (max 5)
    sentiment_summary: Dict[str, int]  # counts of good/bad/neutral in recent reviews
    sentiment_distribution: Dict[str, float]  # percentages in recent reviews
    classified_reviews: List[ReviewSentiment]  # The 5 most recent reviews

# API Endpoints

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Agent Finder API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "recommendations": "/recommend",
            "agent_details": "/agents/{agent_id}",
            "agent_reviews_sentiment": "/agents/{agent_id}/reviews/sentiment",
            "stats": "/stats",
            "health": "/health",
            "train": "/train"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global agent_finder
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_trained": agent_finder is not None and agent_finder.is_trained if agent_finder else False,
        "model_ready": agent_finder is not None
    }

@app.get("/stats")
async def get_system_stats():
    """Get system statistics and model information."""
    global agent_finder
    
    if agent_finder is None:
        raise HTTPException(status_code=404, detail="Agent Finder not initialized")
    
    return agent_finder.get_system_stats()

@app.post("/train")
async def train_system(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train or retrain the Agent Finder system."""
    global agent_finder
    
    # Initialize if not exists
    if agent_finder is None:
        agents_file = os.getenv("AGENTS_FILE", "agents.csv")
        reviews_file = os.getenv("REVIEWS_FILE", "reviews.csv")
        
        if not os.path.exists(agents_file) or not os.path.exists(reviews_file):
            raise HTTPException(
                status_code=400, 
                detail=f"Data files not found: {agents_file}, {reviews_file}"
            )
        
        agent_finder = AgentFinder(agents_file, reviews_file)
    
    try:
        # Train in background for large datasets
        if request.use_cache:
            # Quick training with cache
            training_result = agent_finder.train_system(
                use_cache=request.use_cache,
                save_cache=request.save_cache
            )
            return {
                "message": "Training completed",
                "result": training_result
            }
        else:
            # Start background training
            background_tasks.add_task(
                _train_in_background, 
                request.use_cache, 
                request.save_cache
            )
            return {
                "message": "Training started in background",
                "status": "training"
            }
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_agents(request: RecommendationRequest):
    """Get personalized agent recommendations."""
    global agent_finder
    
    if agent_finder is None or not agent_finder.is_trained:
        raise HTTPException(
            status_code=404, 
            detail="Agent Finder not trained. Please train the system first using /train endpoint."
        )
    
    try:
        # Convert request to internal format
        user_preferences = {
            'responsiveness': request.preferences.responsiveness,
            'negotiation': request.preferences.negotiation,
            'professionalism': request.preferences.professionalism,
            'market_expertise': request.preferences.market_expertise
        }
        
        user_filters = {
            k: v for k, v in request.filters.dict().items() 
            if v is not None
        }
        
        # Get recommendations
        results = agent_finder.recommend_agents(
            user_preferences=user_preferences,
            user_filters=user_filters,
            top_k=request.top_k,
            include_explanations=request.include_explanations
        )
        
        return RecommendationResponse(**results)
    
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/agents/{agent_id}")
async def get_agent_details(agent_id: int):
    """Get detailed information about a specific agent."""
    global agent_finder
    
    if agent_finder is None or not agent_finder.is_trained:
        raise HTTPException(status_code=404, detail="Agent Finder not trained")
    
    # Find agent in the dataset
    agents_df = agent_finder.agents_skill_df
    agent_data = agents_df[agents_df['advertiser_id'] == agent_id]
    
    if len(agent_data) == 0:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = agent_data.iloc[0]
    
    # Get agent reviews
    agent_reviews = agent_finder.reviews_df[
        agent_finder.reviews_df['advertiser_id'] == agent_id
    ]
    
    # Format detailed response
    return {
        "agent_id": int(agent['advertiser_id']),
        "profile": {
            "name": agent.get('full_name', 'Unknown'),
            "state": agent.get('state', ''),
            "city": agent.get('agent_base_city', ''),
            "rating": float(agent.get('agent_rating', 0.0)),
            "review_count": int(agent.get('review_count_actual', 0)),
            "experience_years": float(agent.get('experience_years', 0.0)),
            "specializations": agent.get('specializations', ''),
            "languages": str(agent.get('languages', 'English')) if agent.get('languages') is not None and str(agent.get('languages')) != 'nan' else 'English',
            "agent_type": str(agent.get('agent_type', '')) if agent.get('agent_type') is not None and str(agent.get('agent_type')) != 'nan' else '',
            "office_name": str(agent.get('office_name', '')) if agent.get('office_name') is not None and str(agent.get('office_name')) != 'nan' else '',
            "phone": str(agent.get('phone_primary', '')) if agent.get('phone_primary') is not None and str(agent.get('phone_primary')) != 'nan' else '',
            "website": str(agent.get('agent_website', '')) if agent.get('agent_website') is not None and str(agent.get('agent_website')) != 'nan' else '',
            "bio": str(agent.get('agent_bio', '')) if agent.get('agent_bio') is not None and str(agent.get('agent_bio')) != 'nan' else ''
        },
        "metrics": {
            "responsiveness": float(agent.get('skill_responsiveness', 0.0)),
            "negotiation": float(agent.get('skill_negotiation', 0.0)),
            "professionalism": float(agent.get('skill_professionalism', 0.0)),
            "market_expertise": float(agent.get('skill_market_expertise', 0.0)),
            "q_prior": float(agent.get('q_prior', 0.0)),
            "confidence_score": float(agent.get('confidence_score', 0.0))
        },
        "reviews": {
            "total_count": len(agent_reviews),
            "recent_reviews": agent_reviews.head(5).to_dict('records') if len(agent_reviews) > 0 else []
        }
    }

@app.get("/agents/{agent_id}/reviews/sentiment", response_model=ReviewSentimentResponse)
async def get_agent_reviews_sentiment(agent_id: int):
    """
    Get agent reviews with sentiment analysis.
    
    Returns:
    - Sentiment summary calculated from ALL reviews for the agent
    - Details for the 5 most recent reviews only
    - Sentiment distribution percentages based on ALL reviews
    """
    global agent_finder, sentiment_analyzer
    
    if agent_finder is None or not agent_finder.is_trained:
        raise HTTPException(status_code=404, detail="Agent Finder not trained")
    
    # Initialize sentiment analyzer if not loaded
    if sentiment_analyzer is None:
        try:
            logger.info("Loading sentiment analysis model...")
            sentiment_analyzer = _initialize_sentiment_analyzer()
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {str(e)}")
            raise HTTPException(status_code=500, detail="Sentiment analysis model not available")
    
    # Find agent in the dataset
    agents_df = agent_finder.agents_skill_df
    agent_data = agents_df[agents_df['advertiser_id'] == agent_id]
    
    if len(agent_data) == 0:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = agent_data.iloc[0]
    agent_name = agent.get('full_name', f'Agent {agent_id}')
    
    # Get ALL agent reviews first
    all_agent_reviews = agent_finder.reviews_df[
        agent_finder.reviews_df['advertiser_id'] == agent_id
    ].copy()
    
    if len(all_agent_reviews) == 0:
        raise HTTPException(status_code=404, detail=f"No reviews found for agent {agent_id}")
    
    # Calculate sentiment summary from ALL reviews
    logger.info(f"Analyzing sentiment for {len(all_agent_reviews)} total reviews for agent {agent_id}")
    overall_sentiment_counts = {'good': 0, 'bad': 0, 'neutral': 0}
    
    for _, review in all_agent_reviews.iterrows():
        try:
            review_text = str(review.get('review_comment', ''))
            
            # Skip empty reviews
            if not review_text or review_text.lower() in ['nan', 'none', '']:
                overall_sentiment_counts['neutral'] += 1
                continue
                
            # Classify sentiment using both ML model and rating
            sentiment, confidence = _classify_review_sentiment(
                review_text, 
                review.get('review_rating', 3.0),
                sentiment_analyzer
            )
            
            # Count sentiments for overall summary
            overall_sentiment_counts[sentiment] += 1
            
        except Exception as e:
            logger.warning(f"Failed to process review {review.get('review_id', 'unknown')} for summary: {str(e)}")
            overall_sentiment_counts['neutral'] += 1
            continue
    
    # Now get the 5 most recent reviews for detailed response
    all_agent_reviews['review_created_date'] = pd.to_datetime(all_agent_reviews['review_created_date'], errors='coerce')
    recent_reviews = all_agent_reviews.sort_values('review_created_date', ascending=False, na_position='last').head(5)
    
    logger.info(f"Returning {len(recent_reviews)} most recent reviews for agent {agent_id}")
    
    # Classify the 5 most recent reviews for detailed response
    classified_reviews = []
    
    for _, review in recent_reviews.iterrows():
        try:
            review_text = str(review.get('review_comment', ''))
            
            # Skip empty reviews
            if not review_text or review_text.lower() in ['nan', 'none', '']:
                continue
                
            # Classify sentiment using both ML model and rating
            sentiment, confidence = _classify_review_sentiment(
                review_text, 
                review.get('review_rating', 3.0),
                sentiment_analyzer
            )
            
            # Format review
            classified_review = ReviewSentiment(
                review_id=str(review.get('review_id', '')),
                review_text=review_text,
                review_rating=float(review.get('review_rating', 0.0)),
                sentiment=sentiment,
                sentiment_score=confidence,
                review_date=str(review.get('review_created_date', '')),
                reviewer_role=str(review.get('reviewer_role', 'UNKNOWN')),
                sub_scores={
                    'responsiveness': float(review['sub_responsiveness']) if pd.notna(review.get('sub_responsiveness')) else None,
                    'negotiation': float(review['sub_negotiation']) if pd.notna(review.get('sub_negotiation')) else None,
                    'professionalism': float(review['sub_professionalism']) if pd.notna(review.get('sub_professionalism')) else None,
                    'market_expertise': float(review['sub_market_expertise']) if pd.notna(review.get('sub_market_expertise')) else None
                }
            )
            
            classified_reviews.append(classified_review)
            
        except Exception as e:
            logger.warning(f"Failed to process review {review.get('review_id', 'unknown')}: {str(e)}")
            continue
    
    # Calculate sentiment distribution from overall summary
    total_classified = sum(overall_sentiment_counts.values())
    sentiment_distribution = {}
    if total_classified > 0:
        sentiment_distribution = {
            sentiment: count / total_classified 
            for sentiment, count in overall_sentiment_counts.items()
        }
    else:
        sentiment_distribution = {'good': 0.0, 'bad': 0.0, 'neutral': 0.0}
    
    # Sort reviews by date (most recent first)
    classified_reviews.sort(
        key=lambda x: x.review_date if x.review_date else '1900-01-01', 
        reverse=True
    )
    
    return ReviewSentimentResponse(
        agent_id=agent_id,
        agent_name=agent_name,
        total_reviews=len(all_agent_reviews),  # Total reviews for this agent
        recent_reviews_count=len(classified_reviews),  # Number of recent reviews returned
        sentiment_summary=overall_sentiment_counts,  # Summary from ALL reviews
        sentiment_distribution=sentiment_distribution,  # Distribution from ALL reviews
        classified_reviews=classified_reviews  # Only the 5 most recent
    )

def _initialize_sentiment_analyzer():
    """Initialize the sentiment analysis model."""
    try:
        # Use a lightweight, fast sentiment model
        # You can replace this with a more sophisticated model if needed
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # Check if GPU is available
        device = 0 if torch.cuda.is_available() else -1
        
        analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            return_all_scores=True
        )
        
        logger.info(f"Sentiment analyzer loaded: {model_name} (device: {'GPU' if device >= 0 else 'CPU'})")
        return analyzer
        
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {str(e)}")
        # Fallback to a simpler model
        try:
            analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,  # Force CPU for fallback
                return_all_scores=True
            )
            logger.info("Loaded fallback sentiment model: DistilBERT")
            return analyzer
        except Exception as fallback_error:
            logger.error(f"Fallback model also failed: {str(fallback_error)}")
            raise

def _classify_review_sentiment(review_text: str, review_rating: float, analyzer) -> tuple[str, float]:
    """
    Classify review sentiment using both ML model and rating.
    
    Returns:
        tuple: (sentiment, confidence_score)
            sentiment: 'good', 'bad', or 'neutral'
            confidence_score: float between 0 and 1
    """
    try:
        # Truncate very long reviews to avoid model limits
        max_length = 512
        if len(review_text) > max_length:
            review_text = review_text[:max_length]
        
        # Get ML model prediction
        ml_results = analyzer(review_text)
        
        # Handle different model output formats
        if isinstance(ml_results[0], list):
            ml_scores = ml_results[0]
        else:
            ml_scores = ml_results
            
        # Extract sentiment scores
        ml_sentiment_scores = {}
        for result in ml_scores:
            label = result['label'].lower()
            score = result['score']
            
            # Map different model labels to our categories
            if label in ['positive', 'pos', 'label_2']:
                ml_sentiment_scores['positive'] = score
            elif label in ['negative', 'neg', 'label_0']:
                ml_sentiment_scores['negative'] = score
            elif label in ['neutral', 'label_1']:
                ml_sentiment_scores['neutral'] = score
        
        # Determine ML sentiment
        ml_sentiment = max(ml_sentiment_scores, key=ml_sentiment_scores.get)
        ml_confidence = ml_sentiment_scores[ml_sentiment]
        
        # Rating-based sentiment (as backup/validation)
        if review_rating >= 4.0:
            rating_sentiment = 'positive'
            rating_confidence = min(1.0, (review_rating - 3.0) / 2.0)
        elif review_rating <= 2.0:
            rating_sentiment = 'negative' 
            rating_confidence = min(1.0, (3.0 - review_rating) / 2.0)
        else:
            rating_sentiment = 'neutral'
            rating_confidence = 1.0 - abs(review_rating - 3.0) / 1.0
        
        # Combine ML and rating signals
        # Give more weight to rating for very clear cases, ML for nuanced text
        if abs(review_rating - 3.0) > 1.5:  # Clear rating signal
            final_sentiment = rating_sentiment
            final_confidence = 0.7 * rating_confidence + 0.3 * ml_confidence
        else:  # Ambiguous rating, trust ML more
            final_sentiment = ml_sentiment
            final_confidence = 0.8 * ml_confidence + 0.2 * rating_confidence
        
        # Map to our three categories
        if final_sentiment == 'positive':
            return 'good', final_confidence
        elif final_sentiment == 'negative':
            return 'bad', final_confidence
        else:
            return 'neutral', final_confidence
            
    except Exception as e:
        logger.warning(f"Sentiment classification failed: {str(e)}")
        
        # Fallback to rating-only classification
        if review_rating >= 4.0:
            return 'good', 0.8
        elif review_rating <= 2.0:
            return 'bad', 0.8
        else:
            return 'neutral', 0.6

# Background tasks

async def _train_in_background(use_cache: bool, save_cache: bool):
    """Train system in background task."""
    global agent_finder
    
    try:
        if agent_finder:
            training_result = agent_finder.train_system(
                use_cache=use_cache,
                save_cache=save_cache
            )
            logger.info(f"Background training completed: {training_result['success']}")
    except Exception as e:
        logger.error(f"Background training failed: {str(e)}")

# Startup event

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    global agent_finder, sentiment_analyzer
    
    logger.info("Starting Agent Finder API...")
    
    # Check for environment variables
    agents_file = os.getenv("AGENTS_FILE", "agents.csv")
    reviews_file = os.getenv("REVIEWS_FILE", "reviews.csv")
    
    if os.path.exists(agents_file) and os.path.exists(reviews_file):
        logger.info(f"Data files found: {agents_file}, {reviews_file}")
        agent_finder = AgentFinder(agents_file, reviews_file)
        
        # Try to load cached model
        try:
            training_result = agent_finder.train_system(use_cache=True, save_cache=True)
            logger.info("System trained successfully on startup")
        except Exception as e:
            logger.warning(f"Could not train system on startup: {str(e)}")
            
        # Initialize sentiment analyzer in background
        try:
            logger.info("Initializing sentiment analysis model...")
            sentiment_analyzer = _initialize_sentiment_analyzer()
            logger.info("Sentiment analyzer ready")
        except Exception as e:
            logger.warning(f"Could not initialize sentiment analyzer on startup: {str(e)}")
            logger.info("Sentiment analyzer will be loaded on first request")
    else:
        logger.warning(f"Data files not found: {agents_file}, {reviews_file}")
        logger.info("System will need to be initialized manually via /train endpoint")

# Run with: uvicorn src.api:app --reload --host 0.0.0.0 --port 8003

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
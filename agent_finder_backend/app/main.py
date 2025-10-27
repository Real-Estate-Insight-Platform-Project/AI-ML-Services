from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import (
    AgentSearchRequest,
    AgentSearchResponse,
    AgentRecommendation,
    AgentReviewsResponse,
    CitiesResponse,
    StatesResponse,
    ErrorResponse
)
from app.service import recommendation_service
from config.settings import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Real Estate Agent Recommendation API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Agent Finder API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "agent-finder"
    }


@app.post(
    f"{settings.API_V1_PREFIX}/agents/search",
    response_model=AgentSearchResponse,
    responses={
        200: {"description": "Successful search"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def search_agents(request: AgentSearchRequest):
    """
    Search for real estate agents based on user criteria.
    
    This endpoint finds and ranks agents using a sophisticated algorithm that considers:
    - Buyer/seller fit based on historical performance
    - Recent sales activity and experience
    - Review sentiment and quality scores
    - User-weighted preferences for responsiveness, negotiation, etc.
    - Geographic proximity
    - Property type specialization
    - Additional specializations
    
    The matching score (0-100) reflects how well the agent matches your criteria.
    """
    try:
        logger.info(f"Received search request: {request.dict()}")
        
        # Find agents
        recommendations = recommendation_service.find_agents(request)
        
        logger.info(f"Found {len(recommendations)} matching agents")
        
        # Prepare response
        response = AgentSearchResponse(
            success=True,
            message=f"Found {len(recommendations)} matching agents",
            total_results=len(recommendations),
            recommendations=recommendations,
            search_params=request.dict()
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )


@app.get(f"{settings.API_V1_PREFIX}/agents/{{agent_id}}")
async def get_agent_details(agent_id: int):
    """
    Get detailed information about a specific agent.
    
    Args:
        agent_id: Agent's advertiser ID
    
    Returns:
        Agent details including scores and statistics
    """
    try:
        from utils.database import db_client
        
        agent = db_client.get_agent_by_id(agent_id)
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID {agent_id} not found"
            )
        
        # Get reviews
        reviews_df = db_client.get_reviews_by_agent(agent_id)
        
        # Add review summary
        agent['reviews_summary'] = {
            'total_reviews': len(reviews_df),
            'recent_reviews': len(reviews_df[reviews_df['days_since_review'] <= 365]) if 'days_since_review' in reviews_df.columns else 0,
            'sentiment_distribution': reviews_df['sentiment'].value_counts().to_dict() if 'sentiment' in reviews_df.columns else {}
        }
        
        return {
            "success": True,
            "agent": agent
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching agent details: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching agent details"
        )


@app.get(f"{settings.API_V1_PREFIX}/stats")
async def get_statistics():
    """
    Get system statistics.
    
    Returns:
        Statistics about agents, reviews, and coverage
    """
    try:
        from utils.database import db_client
        import pandas as pd
        
        agents_df = db_client.get_all_agents()
        reviews_df = db_client.get_all_reviews()
        
        # Calculate statistics
        stats = {
            "total_agents": len(agents_df),
            "total_reviews": len(reviews_df),
            "states_covered": agents_df['state'].nunique(),
            "cities_covered": agents_df['agent_base_city'].nunique(),
            "avg_agent_rating": float(agents_df['agent_rating'].mean()) if 'agent_rating' in agents_df.columns else 0,
            "agents_with_reviews": len(agents_df[agents_df['review_count'] > 0]) if 'review_count' in agents_df.columns else 0,
            "sentiment_distribution": reviews_df['sentiment'].value_counts().to_dict() if 'sentiment' in reviews_df.columns else {}
        }
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching statistics"
        )


@app.post(
    f"{settings.API_V1_PREFIX}/agents/reviews",
    response_model=AgentReviewsResponse,
    responses={
        200: {"description": "Successful retrieval of agent reviews"},
        404: {"model": ErrorResponse, "description": "Agent not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_agent_reviews(agent_id: int):
    """
    POST: Get top 5 recent reviews for a specific agent along with review counts.
    
    Args:
        agent_id: Agent's advertiser ID (path parameter)
    
    Returns:
        Agent's review counts and top 5 recent reviews
    """
    try:
        from utils.database import db_client
        
        # Get agent details for review counts
        agent = db_client.get_agent_by_id(agent_id)
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID {agent_id} not found"
            )
        
        # Get top 5 recent reviews for this agent
        reviews = db_client.get_recent_reviews_by_agent(agent_id, limit=5)
        
        # Extract review counts from agent data
        review_counts = {
            'total_review_count': agent.get('review_count', 0),
            'positive_review_count': agent.get('positive_review_count', 0),
            'negative_review_count': agent.get('negative_review_count', 0),
            'neutral_review_count': agent.get('neutral_review_count', 0)
        }
        
        return {
            "success": True,
            "agent_id": agent_id,
            "agent_name": agent.get('full_name', 'Unknown'),
            "review_counts": review_counts,
            "recent_reviews": reviews
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching agent reviews: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching agent reviews"
        )


@app.get(
    f"{settings.API_V1_PREFIX}/locations/cities",
    response_model=CitiesResponse,
    responses={
        200: {"description": "Successful retrieval of cities"},
        404: {"model": ErrorResponse, "description": "State not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_cities_by_state(state_name: str):
    """
    GET: Get all cities for a specific state.
    
    Args:
        state_name: State name as query parameter (e.g., "California", "Texas")
    
    Returns:
        List of cities in the specified state
    """
    try:
        from utils.database import db_client

        if not state_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="state_name query parameter is required"
            )
        
        # Get cities from uszips table for the specified state
        cities = db_client.get_cities_by_state(state_name)
        
        if not cities:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No cities found for state: {state_name}"
            )
        
        return {
            "success": True,
            "state_name": state_name,
            "total_cities": len(cities),
            "cities": cities
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching cities: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching cities"
        )


@app.get(
    f"{settings.API_V1_PREFIX}/locations/states",
    response_model=StatesResponse,
    responses={
        200: {"description": "Successful retrieval of states"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_all_states():
    """
    Get all available states.
    
    Returns:
        List of all states available in the database
    """
    try:
        from utils.database import db_client
        
        # Get all unique states from uszips table
        states = db_client.get_all_states()
        
        if not states:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No states found in database"
            )
        
        return {
            "success": True,
            "total_states": len(states),
            "states": states
        }
        
    except Exception as e:
        logger.error(f"Error fetching states: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching states"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "details": str(exc) if settings.DEBUG else None
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=True
    )
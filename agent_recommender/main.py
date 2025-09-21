# agent_recommender/main.py
import os
import sys
import logging
import warnings
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Suppress verbose logging and warnings during startup
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
logging.getLogger('sklearn').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Import the recommender system after setting up logging
try:
    from recommender import AgentRecommenderSystem
    AGENT_RECOMMENDER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import AgentRecommenderSystem: {e}")
    AGENT_RECOMMENDER_AVAILABLE = False

load_dotenv()

# Configuration
AGENT_DATA_PATH = "../agent_data/statewise_data"

# Pydantic models for request validation
class AgentRecommendationRequest(BaseModel):
    regions: List[str] = Field(..., description="List of target cities/regions")
    budget: float = Field(..., gt=0, description="Budget amount in USD")
    property_types: List[str] = Field(..., description="List of desired property types")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations to return")
    model_type: str = Field("baseline", description="Model type: baseline, ml, or ensemble")
    explain: bool = Field(False, description="Include detailed explanations")

class AgentCompareRequest(BaseModel):
    regions: List[str] = Field(..., description="List of target cities/regions")
    budget: float = Field(..., gt=0, description="Budget amount in USD")
    property_types: List[str] = Field(..., description="List of desired property types")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations to return per model")

# ---------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------

app = FastAPI(
    title="Agent Recommender API",
    description="Real Estate Agent Recommendation System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1500)

# ---------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    """Initialize the Agent Recommender System."""
    if AGENT_RECOMMENDER_AVAILABLE:
        try:
            # Suppress additional logging during initialization
            logging.getLogger('recommender').setLevel(logging.WARNING)
            logging.getLogger('utils.data_preprocessing').setLevel(logging.WARNING)
            logging.getLogger('models.baseline_scorer').setLevel(logging.WARNING)
            logging.getLogger('models.ml_ranker').setLevel(logging.WARNING)
            
            app.state.agent_recommender = AgentRecommenderSystem(AGENT_DATA_PATH)
            app.state.agent_recommender.initialize()
            print("✅ Agent Recommender System initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Agent Recommender: {e}")
            app.state.agent_recommender = None
    else:
        app.state.agent_recommender = None

@app.on_event("shutdown")
async def _shutdown():
    """Clean up resources."""
    # No special cleanup needed for the current implementation
    pass

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _to_jsonable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    try:
        return float(obj)  # Handle numpy types, etc.
    except Exception:
        return str(obj)

# ---------------------------------------------------------------------
# Agent Recommender API Routes
# ---------------------------------------------------------------------

@app.post("/recommend", tags=["recommendations"])
async def recommend_agents(request: AgentRecommendationRequest):
    """
    Get real estate agent recommendations based on user preferences.
    
    - **regions**: List of target cities/regions
    - **budget**: Budget amount in USD
    - **property_types**: List of desired property types
    - **top_k**: Number of recommendations to return (default: 5)
    - **model_type**: Model to use - 'baseline', 'ml', or 'ensemble' (default: 'baseline')
    - **explain**: Include detailed explanations (default: False)
    """
    if not app.state.agent_recommender:
        raise HTTPException(
            status_code=503, 
            detail="Agent Recommender service not available"
        )
    
    try:
        # Validate model type
        if request.model_type not in ['baseline', 'ml', 'ensemble']:
            raise HTTPException(
                status_code=400,
                detail="model_type must be one of: baseline, ml, ensemble"
            )
        
        # Create query
        user_query = {
            'regions': request.regions,
            'budget': request.budget,
            'property_types': request.property_types
        }
        
        # Get recommendations
        result = app.state.agent_recommender.recommend(
            user_query,
            model_type=request.model_type,
            top_k=request.top_k,
            explain=request.explain
        )
        
        return JSONResponse(
            _to_jsonable(result),
            headers={"Cache-Control": "public, max-age=300"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/agents/{agent_id}", tags=["agents"])
async def get_agent_details(agent_id: int):
    """
    Get detailed information about a specific real estate agent.
    
    - **agent_id**: Unique agent identifier
    """
    if not app.state.agent_recommender:
        raise HTTPException(
            status_code=503,
            detail="Agent Recommender service not available"
        )
    
    try:
        details = app.state.agent_recommender.get_agent_details(agent_id)
        
        if 'error' in details:
            raise HTTPException(status_code=404, detail=details['error'])
        
        return JSONResponse(
            _to_jsonable(details),
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/compare", tags=["recommendations"])
async def compare_agent_models(request: AgentCompareRequest):
    """
    Compare recommendations from different models (baseline vs ML).
    
    - **regions**: List of target cities/regions
    - **budget**: Budget amount in USD
    - **property_types**: List of desired property types
    - **top_k**: Number of recommendations to return per model (default: 5)
    """
    if not app.state.agent_recommender:
        raise HTTPException(
            status_code=503,
            detail="Agent Recommender service not available"
        )
    
    try:
        # Create query
        user_query = {
            'regions': request.regions,
            'budget': request.budget,
            'property_types': request.property_types
        }
        
        # Compare models
        comparison = app.state.agent_recommender.compare_models(
            user_query,
            top_k=request.top_k
        )
        
        return JSONResponse(
            _to_jsonable(comparison),
            headers={"Cache-Control": "public, max-age=300"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/stats", tags=["system"])
async def get_agent_system_stats():
    """
    Get statistics about the agent recommender system.
    
    Returns information about:
    - Total number of agents
    - States and markets covered
    - Data quality metrics
    - System performance
    """
    if not app.state.agent_recommender:
        raise HTTPException(
            status_code=503,
            detail="Agent Recommender service not available"
        )
    
    try:
        stats = app.state.agent_recommender.get_system_stats()
        
        return JSONResponse(
            _to_jsonable(stats),
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/health", tags=["system"])
async def agent_health_check():
    """
    Health check for the agent recommender system.
    """
    if not app.state.agent_recommender:
        return JSONResponse(
            {
                "status": "unavailable",
                "message": "Agent Recommender service not initialized",
                "healthy": False
            },
            status_code=503
        )
    
    try:
        # Quick system check
        stats = app.state.agent_recommender.get_system_stats()
        
        return JSONResponse({
            "status": "healthy",
            "message": "Agent Recommender service operational",
            "healthy": True,
            "total_agents": stats.get('total_agents', 0),
            "unique_states": stats.get('unique_states', 0),
            "unique_markets": stats.get('unique_markets', 0)
        })
        
    except Exception as e:
        return JSONResponse(
            {
                "status": "unhealthy",
                "message": f"Agent Recommender service error: {str(e)}",
                "healthy": False
            },
            status_code=500
        )

# ---------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------

@app.get("/", tags=["system"])
async def root():
    """API information and available endpoints."""
    agent_status = "available" if app.state.agent_recommender else "unavailable"
    
    return {
        "ok": True,
        "service": "agent-recommender",
        "message": "Real Estate Agent Recommendation System",
        "status": agent_status,
        "endpoints": {
            "recommend": "/recommend",
            "agent_details": "/agents/{id}",
            "compare_models": "/compare", 
            "system_stats": "/stats",
            "health_check": "/health"
        },
        "documentation": "/docs"
    }

# ---------------------------------------------------------------------
# Development server
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
"""
Main FastAPI Application
Production-ready agentic real estate assistant with comprehensive features
"""
import uuid
import time
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
from sqlalchemy import text

from src.config import settings
from src.workflows.react_agent import agent
from src.utils.session_manager import session_manager
from src.tools.tools import db_clients

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.log_format == "json" 
        else structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics - handle duplicates gracefully
def create_or_get_counter(name, description, labels, registry=REGISTRY):
    """Create a counter or return existing one"""
    try:
        return Counter(name, description, labels, registry=registry)
    except ValueError:
        # Metric already exists, get it from registry
        for collector in registry._collector_to_names.keys():
            if hasattr(collector, '_name') and collector._name == name:
                return collector
        raise

def create_or_get_histogram(name, description, labels, registry=REGISTRY):
    """Create a histogram or return existing one"""
    try:
        return Histogram(name, description, labels, registry=registry)
    except ValueError:
        # Metric already exists, get it from registry
        for collector in registry._collector_to_names.keys():
            if hasattr(collector, '_name') and collector._name == name:
                return collector
        raise

REQUEST_COUNT = create_or_get_counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = create_or_get_histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
CHAT_REQUESTS = create_or_get_counter('chat_requests_total', 'Total chat requests', ['intent'])
TOOL_EXECUTIONS = create_or_get_counter('tool_executions_total', 'Total tool executions', ['tool_name', 'status'])

# Rate limiter - DISABLED
# limiter = Limiter(key_func=get_remote_address)


# ==================== PYDANTIC MODELS ====================

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message", min_length=1, max_length=5000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Show me 3-bedroom houses in Boston under $800k",
                "session_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    session_id: str
    timestamp: str
    intent: Optional[str] = None
    success: bool = True
    metadata: Optional[Dict[str, Any]] = None


class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str
    exists: bool
    message_count: Optional[int] = None
    created_at: Optional[str] = None
    last_activity: Optional[str] = None
    ttl_seconds: Optional[int] = None
    ttl_minutes: Optional[float] = None


# ==================== LIFESPAN MANAGEMENT ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("application_startup", 
               app_name=settings.app_name,
               version=settings.app_version,
               port=settings.app_port)
    
    # Verify database connections
    try:
        with db_clients.supabase_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM properties"))
            count = result.fetchone()[0]
            logger.info("supabase_connected", property_count=count)
    except Exception as e:
        logger.error("supabase_connection_failed", error=str(e))
    
    try:
        test_query = f"SELECT 1 FROM `{settings.bigquery_dataset_full}.state_market` LIMIT 1"
        db_clients.bigquery_client.query(test_query).result()
        logger.info("bigquery_connected", project=settings.bq_project_id)
    except Exception as e:
        logger.error("bigquery_connection_failed", error=str(e))
    
    # Check Redis
    redis_health = session_manager.health_check()
    logger.info("redis_status", **redis_health)
    
    yield
    
    # Cleanup
    logger.info("application_shutdown")
    if db_clients._supabase_engine:
        db_clients._supabase_engine.dispose()
    if db_clients._bigquery_client:
        db_clients._bigquery_client.close()


# ==================== APPLICATION SETUP ====================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready agentic real estate assistant with ReAct workflow, multi-database support, and comprehensive market intelligence",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiting - DISABLED
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== MIDDLEWARE ====================

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to state
    request.state.request_id = request_id
    
    logger.info("request_started",
               request_id=request_id,
               method=request.method,
               path=request.url.path,
               client=request.client.host if request.client else "unknown")
    
    try:
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        # Update metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        logger.info("request_completed",
                   request_id=request_id,
                   status_code=response.status_code,
                   duration_ms=round(duration * 1000, 2))
        
        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("request_error",
                    request_id=request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        raise


# ==================== HEALTH & MONITORING ====================

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.app_version,
        "services": {}
    }
    
    # Check Supabase
    try:
        with db_clients.supabase_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health_status["services"]["supabase"] = {"status": "healthy"}
    except Exception as e:
        health_status["services"]["supabase"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check BigQuery
    try:
        test_query = f"SELECT 1 FROM `{settings.bigquery_dataset_full}.state_market` LIMIT 1"
        db_clients.bigquery_client.query(test_query).result()
        health_status["services"]["bigquery"] = {"status": "healthy", "project": settings.bq_project_id}
    except Exception as e:
        health_status["services"]["bigquery"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check Redis
    redis_health = session_manager.health_check()
    health_status["services"]["redis"] = redis_health
    
    # Check Agent
    health_status["services"]["agent"] = {"status": "healthy", "model": settings.gemini_flash_model}
    
    return health_status


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats", tags=["Monitoring"])
async def get_stats():
    """Get application statistics"""
    redis_stats = session_manager.get_stats()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "version": settings.app_version,
        "redis": redis_stats,
        "config": {
            "session_ttl_minutes": settings.redis_session_ttl // 60,
            "cache_ttl_seconds": settings.redis_cache_ttl,
            "max_query_results": settings.max_query_results,
            "rate_limit_enabled": False  # Rate limiting disabled
        }
    }


# ==================== CHAT ENDPOINTS ====================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
# @limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}second")  # RATE LIMITING DISABLED
async def chat(request: Request, chat_request: ChatRequest):
    """
    Send a message to the agentic assistant
    
    The assistant can:
    - Search properties in Supabase
    - Analyze market trends in BigQuery
    - Geocode locations and find nearby properties
    - Assess disaster risks
    - Provide investment recommendations
    - Search the web for current information
    """
    # Generate or use provided session ID
    session_id = chat_request.session_id or str(uuid.uuid4())
    message = chat_request.message
    
    logger.info("chat_request_received", session_id=session_id, message_length=len(message))
    
    try:
        # Check cache for similar queries
        cached_result = session_manager.get_cached_query(message)
        if cached_result:
            logger.info("cache_hit", session_id=session_id)
            return ChatResponse(
                response=cached_result.get("response", ""),
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                intent=cached_result.get("intent"),
                success=True,
                metadata={"cached": True}
            )
        
        # Get conversation history
        history = session_manager.get_conversation(session_id)
        
        # Add user message to history
        session_manager.add_message(session_id, "user", message)
        
        # Run agent
        result = agent.run(
            query=message,
            session_id=session_id,
            history=history
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Agent execution failed")
            )
        
        response_text = result.get("response", "")
        intent = result.get("intent", "")
        
        # Add assistant response to history
        session_manager.add_message(session_id, "assistant", response_text, {
            "intent": intent,
            "iterations": result.get("iterations", 0)
        })
        
        # Cache result
        session_manager.cache_query_result(message, {
            "response": response_text,
            "intent": intent
        })
        
        # Update metrics
        CHAT_REQUESTS.labels(intent=intent).inc()
        
        # Extend session TTL
        session_manager.extend_session(session_id)
        
        logger.info("chat_request_completed", 
                   session_id=session_id,
                   intent=intent,
                   response_length=len(response_text))
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            intent=intent,
            success=True,
            metadata={
                "iterations": result.get("iterations", 0),
                "context": result.get("context", {})
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("chat_request_error", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )


# ==================== SESSION MANAGEMENT ====================

@app.get("/session/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def get_session(session_id: str):
    """Get session information and history"""
    info = session_manager.get_session_info(session_id)
    
    if not info.get("exists"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    return SessionInfo(session_id=session_id, **info)


@app.delete("/session/{session_id}", tags=["Sessions"])
async def clear_session(session_id: str):
    """Clear session history"""
    success = session_manager.clear_conversation(session_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear session"
        )
    
    return {"success": True, "message": f"Session {session_id} cleared"}


@app.post("/session/{session_id}/extend", tags=["Sessions"])
async def extend_session_ttl(session_id: str):
    """Extend session TTL"""
    success = session_manager.extend_session(session_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    return {
        "success": True,
        "message": f"Session TTL extended",
        "ttl_seconds": settings.redis_session_ttl
    }


@app.get("/sessions", tags=["Sessions"])
async def list_sessions():
    """List all active sessions"""
    sessions = session_manager.get_all_sessions()
    
    return {
        "active_sessions": len(sessions),
        "sessions": sessions[:100]  # Limit to 100 for performance
    }


# ==================== ROOT & INFO ====================

@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Production-ready agentic real estate assistant",
        "features": [
            "ReAct agent with LangGraph workflow",
            "Gemini 2.5 Flash intent classification",
            "Multi-database support (Supabase + BigQuery)",
            "Redis session management with caching",
            "Geocoding and spatial queries",
            "Market analysis and predictions",
            "Risk assessment integration",
            "Web search capability",
            "Monitoring and observability",
            "Unlimited requests (rate limiting disabled)"
        ],
        "endpoints": {
            "POST /chat": "Send message to assistant",
            "GET /session/{session_id}": "Get session info",
            "DELETE /session/{session_id}": "Clear session",
            "POST /session/{session_id}/extend": "Extend session TTL",
            "GET /sessions": "List active sessions",
            "GET /health": "Health check",
            "GET /metrics": "Prometheus metrics",
            "GET /stats": "Application statistics",
            "GET /docs": "API documentation",
        },
        "agent_capabilities": {
            "property_search": "Search properties by location, price, features",
            "market_analysis": "Analyze trends, statistics, predictions",
            "investment_advice": "Combine risk, market data for recommendations",
            "geocoding": "Convert locations to coordinates",
            "spatial_queries": "Find properties near landmarks",
            "risk_assessment": "FEMA disaster risk analysis",
            "web_search": "Current market information"
        },
        "session_config": {
            "backend": "Redis" if session_manager.available else "In-Memory",
            "ttl_minutes": settings.redis_session_ttl // 60,
            "cache_enabled": session_manager.available
        }
    }


# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning("http_exception",
                  request_id=getattr(request.state, "request_id", "unknown"),
                  status_code=exc.status_code,
                  detail=exc.detail)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error("unhandled_exception",
                request_id=getattr(request.state, "request_id", "unknown"),
                error=str(exc))
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug_mode else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    print(f"\n{'='*80}")
    print(f"🤖 {settings.app_name.upper()}")
    print(f"{'='*80}")
    print(f"Version: {settings.app_version}")
    print(f"Port: {settings.app_port}")
    print(f"Redis: {'Connected' if session_manager.available else 'Fallback Mode'}")
    print(f"Debug: {settings.debug_mode}")
    print(f"{'='*80}\n")
    
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.lower(),
        reload=settings.debug_mode
    )

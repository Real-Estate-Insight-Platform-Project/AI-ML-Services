# Agentic Real Estate Assistant

> **Production-ready AI-powered real estate assistant with ReAct agent workflow, multi-database support, and comprehensive market intelligence.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🌟 Features

### Core Capabilities
- **🤖 ReAct Agent Architecture** - Reasoning + Acting pattern with LangGraph orchestration
- **🎯 Intent Classification** - Gemini 2.5 Flash for intelligent query understanding
- **🗄️ Multi-Database Support** - Supabase (PostgreSQL/PostGIS) + Google BigQuery
- **💾 Redis Session Management** - Persistent conversations with caching layer
- **📍 Geocoding Service** - Convert locations to coordinates (US Census + Nominatim)
- **🔍 Web Search Integration** - Current market information retrieval
- **📊 Investment Analysis** - Combine risk data, market trends, and predictions
- **🏠 Property Search** - Advanced filtering with spatial queries
- **📈 Market Analytics** - Historical trends and future predictions
- **⚠️ Risk Assessment** - FEMA National Risk Index integration
- **🚀 Production Ready** - Rate limiting, monitoring, observability

### Technical Features
- **Structured Logging** - JSON-formatted logs with structlog
- **Prometheus Metrics** - Request tracking and performance monitoring
- **Rate Limiting** - Configurable request throttling
- **Health Checks** - Comprehensive service monitoring
- **Docker Support** - Container-ready with docker-compose
- **Comprehensive Tests** - Unit, integration, and performance tests
- **Type Safety** - Full Pydantic validation
- **Error Handling** - Graceful degradation and fallbacks

## 📋 Table of Contents
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Deployment](#deployment)
- [Development](#development)
- [Testing](#testing)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Routing   │  │ Middleware  │  │  Monitoring │         │
│  │  & Handlers │  │   CORS/     │  │ Prometheus  │         │
│  │             │  │ Rate Limit  │  │   Metrics   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph ReAct Agent Workflow                  │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐│
│  │   Intent     │ ──► │   Reasoning  │──► │    Action    ││
│  │Classification│     │     Node     │    │Execution Node││
│  │(Gemini Flash)│     │(Gemini Pro)  │    │(Tool Calls)  ││
│  └──────────────┘     └──────────────┘    └──────────────┘│
│         │                    │                     │        │
│         └────────────────────┴─────────────────────┘        │
│                            │                                │
│                    ┌──────────────┐                         │
│                    │   Response   │                         │
│                    │  Formatter   │                         │
│                    │(Gemini Flash)│                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Supabase  │    │  BigQuery   │    │   Redis     │
│   SQL Agent │    │  SQL Agent  │    │   Session   │
│             │    │             │    │  Management │
│ • properties│    │ • county_   │    │             │
│ • nri_      │    │   market    │    │ • History   │
│   counties  │    │ • county_   │    │ • Caching   │
│ • uszips    │    │   predictions│   │ • Context   │
│ • us_       │    │ • state_    │    │             │
│   counties  │    │   market    │    │             │
│   (PostGIS) │    │ • lookups   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Agent Tools
1. **query_supabase** - Property listings, risk data, ZIP codes
2. **query_bigquery** - Market statistics, trends, predictions
3. **geocode_location** - Location to coordinates conversion
4. **calculate_distance** - Distance calculations for spatial queries
5. **search_web** - Web search for current information
6. **analyze_investment_potential** - Investment recommendations
7. **find_real_estate_agents** - Agent finder portal redirect

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Redis (for session management)
- Google Cloud credentials (BigQuery)
- Supabase database access
- Google API key (Gemini)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd agentic_real_estate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy template
cp .env.template .env

# Edit .env with your credentials
nano .env
```

### 3. Add Service Keys
```bash
# Place your Google Cloud service account key
cp /path/to/your/service_keys.json ./service_keys.json
```

### 4. Run Application
```bash
# Development mode
python src/main.py

# Production mode with uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8005 --workers 4
```

### 5. Test the API
```bash
# Health check
curl http://localhost:8005/health

# Send a chat message
curl -X POST http://localhost:8005/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me 3-bedroom houses in Boston under $800k",
    "session_id": "test-session-123"
  }'
```

## 📦 Installation

### Using pip
```bash
pip install -r requirements.txt
```

### Using Docker
```bash
# Build image
docker build -t real-estate-assistant .

# Run container
docker run -d \
  --name real-estate-assistant \
  -p 8005:8005 \
  --env-file .env \
  -v $(pwd)/service_keys.json:/app/service_keys.json:ro \
  real-estate-assistant
```

### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# Start with monitoring stack
docker-compose --profile monitoring up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## ⚙️ Configuration

### Environment Variables

#### Required
```bash
GOOGLE_API_KEY=your_gemini_api_key
SQL_AGENT_DATABASE_URL=postgresql://user:pass@host:port/db
BQ_PROJECT_ID=your-bigquery-project
REDIS_URL=redis://localhost:6379/0
```

#### Optional
```bash
# Models
GEMINI_FLASH_MODEL=gemini-2.5-flash

# Application
APP_PORT=8005
DEBUG_MODE=false

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Session Management
REDIS_SESSION_TTL=3600
REDIS_CACHE_TTL=300

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Database Schema

#### Supabase Tables
- **properties** - Property listings with location and risk data
- **nri_counties** - FEMA disaster risk assessment by county
- **uszips** - US ZIP code database with demographics
- **gis.us_counties** - County polygons for spatial queries (PostGIS)

#### BigQuery Tables
- **county_market** - Historical market statistics by county
- **county_predictions** - AI forecasts by county
- **state_market** - State-level market statistics
- **state_predictions** - State-level forecasts
- **county_lookup** - County reference data
- **state_lookup** - State reference data

## 📚 API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8005/docs
- **ReDoc**: http://localhost:8005/redoc

### Main Endpoints

#### POST /chat
Send a message to the assistant.

**Request:**
```json
{
  "message": "Show me properties near Boston University",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "response": "I found several properties near Boston University...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-10-26T10:30:00Z",
  "intent": "PROPERTY_SEARCH",
  "success": true,
  "metadata": {
    "iterations": 3,
    "context": {}
  }
}
```

#### GET /session/{session_id}
Get session information and history.

#### DELETE /session/{session_id}
Clear session history.

#### POST /session/{session_id}/extend
Extend session TTL.

#### GET /sessions
List all active sessions.

#### GET /health
Comprehensive health check.

#### GET /metrics
Prometheus metrics.

#### GET /stats
Application statistics.

## 💡 Usage Examples

### Property Search
```python
import requests

response = requests.post("http://localhost:8005/chat", json={
    "message": "Find 3-bedroom houses in San Francisco under $1.2M with low earthquake risk",
    "session_id": "user-123"
})

print(response.json()["response"])
```

### Market Analysis
```python
response = requests.post("http://localhost:8005/chat", json={
    "message": "What are the market trends in Austin, Texas? Is it a buyer's or seller's market?",
    "session_id": "user-123"
})
```

### Investment Advice
```python
response = requests.post("http://localhost:8005/chat", json={
    "message": "Which counties have the best investment potential with low disaster risk and predicted price appreciation?",
    "session_id": "user-123"
})
```

### Location-Based Search
```python
response = requests.post("http://localhost:8005/chat", json={
    "message": "Show me properties within 10 miles of MIT with parking",
    "session_id": "user-123"
})
```

## 🚢 Deployment

### Production Deployment

#### 1. Docker Deployment
```bash
# Build production image
docker build -t real-estate-assistant:v3.0.0 .

# Run with production settings
docker run -d \
  --name real-estate-assistant \
  --restart unless-stopped \
  -p 8005:8005 \
  --env-file .env.production \
  -v $(pwd)/service_keys.json:/app/service_keys.json:ro \
  -v $(pwd)/logs:/app/logs \
  real-estate-assistant:v3.0.0
```

#### 2. Kubernetes Deployment
```bash
# Create namespace
kubectl create namespace real-estate

# Create secrets
kubectl create secret generic app-secrets \
  --from-env-file=.env.production \
  -n real-estate

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

#### 3. Cloud Run (GCP)
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/real-estate-assistant

# Deploy
gcloud run deploy real-estate-assistant \
  --image gcr.io/PROJECT_ID/real-estate-assistant \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "$(cat .env.production | xargs)"
```

### Scaling Considerations
- **Horizontal Scaling**: Run multiple instances behind load balancer
- **Redis Clustering**: Use Redis Cluster for high availability
- **Database Connection Pooling**: Configured via environment variables
- **Rate Limiting**: Adjust based on traffic patterns
- **Caching**: Leverage Redis cache for frequent queries

## 👨‍💻 Development

### Project Structure
```
agentic_real_estate/
├── src/
│   ├── agents/
│   │   ├── supabase_domain.py    # Supabase SQL agent knowledge
│   │   └── bigquery_domain.py    # BigQuery SQL agent knowledge
│   ├── tools/
│   │   └── tools.py               # Tool implementations
│   ├── workflows/
│   │   └── react_agent.py         # LangGraph ReAct agent
│   ├── utils/
│   │   └── session_manager.py     # Redis session management
│   ├── config.py                   # Configuration management
│   └── main.py                     # FastAPI application
├── tests/
│   └── test_main.py               # Comprehensive tests
├── config/
│   └── prometheus.yml             # Prometheus configuration
├── logs/                          # Application logs
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Multi-container setup
├── requirements.txt               # Python dependencies
├── .env.template                  # Environment template
├── service_keys.json              # GCP credentials
└── README.md                      # This file
```

### Adding New Tools
```python
# In src/tools/tools.py

from langchain_core.tools import tool

@tool
def my_new_tool(param: str) -> Dict[str, Any]:
    """
    Tool description for the agent.
    
    Args:
        param: Parameter description
    
    Returns:
        Result dictionary
    """
    # Implementation
    return {"success": True, "data": "result"}

# Add to TOOLS list
TOOLS = [
    # ... existing tools
    my_new_tool,
]
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/ -v -m "not integration and not performance"

# Integration tests
pytest tests/ -v -m integration

# Performance tests
pytest tests/ -v -m performance
```

### Test Coverage
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load test
locust -f tests/locustfile.py --host=http://localhost:8005
```

## 📊 Monitoring

### Prometheus Metrics
Access metrics at: http://localhost:8005/metrics

Key metrics:
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `chat_requests_total` - Chat requests by intent
- `tool_executions_total` - Tool execution counts

### Grafana Dashboards
1. Start monitoring stack: `docker-compose --profile monitoring up -d`
2. Access Grafana: http://localhost:3000 (admin/admin)
3. Add Prometheus datasource: http://prometheus:9090
4. Import dashboards from `config/grafana/`

### Logging
```bash
# View application logs
tail -f logs/app.log

# Docker logs
docker-compose logs -f app

# Filter by log level
cat logs/app.log | grep ERROR
```

### Health Monitoring
```bash
# Check health endpoint
curl http://localhost:8005/health | jq

# Get statistics
curl http://localhost:8005/stats | jq
```

## 🔧 Troubleshooting

### Common Issues

#### Redis Connection Failed
```bash
# Check Redis is running
redis-cli ping

# Verify Redis URL
echo $REDIS_URL

# Test connection
redis-cli -u $REDIS_URL ping
```

#### BigQuery Authentication Error
```bash
# Verify service account key
cat service_keys.json | jq

# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/service_keys.json"

# Test BigQuery connection
bq ls
```

#### Supabase Connection Error
```bash
# Test database connection
psql "$SQL_AGENT_DATABASE_URL" -c "SELECT 1"

# Check connection string format
echo $SQL_AGENT_DATABASE_URL
```

#### High Memory Usage
- Reduce `DB_POOL_SIZE` and `DB_MAX_OVERFLOW`
- Lower `REDIS_SESSION_TTL` to expire sessions faster
- Adjust `MAX_QUERY_RESULTS` to limit result sets

#### Slow Response Times
- Enable Redis caching
- Increase worker count: `--workers 4`
- Optimize SQL queries with proper indexes
- Use connection pooling

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 📞 Support

- Documentation: http://localhost:8005/docs
- Issues: GitHub Issues
- Email: support@example.com

## 🗺️ Roadmap

### Week 1 ✅
- Core SQL agents (Supabase + BigQuery)
- LangGraph ReAct workflow
- Basic tool implementations
- Session management

### Week 2 🚧
- Enhanced geocoding with fallbacks
- PostGIS spatial queries
- Intelligent query router
- Advanced web search

### Week 3 📋
- Production observability
- Comprehensive testing
- Docker deployment
- CI/CD pipeline
- Performance optimization

### Future Enhancements
- Vector search for similar properties
- Natural language query parsing
- Real-time market alerts
- Multi-language support
- Mobile app integration

---

**Built with ❤️ using FastAPI, LangGraph, and Google Gemini**

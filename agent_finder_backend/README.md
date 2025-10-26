# Agent Finder Backend

A sophisticated real estate agent recommendation system with advanced data science algorithms and FastAPI backend.

## 🎯 Features

### Core Capabilities
- **Intelligent Agent Matching**: Multi-factor scoring algorithm with 20+ features
- **Sentiment Analysis**: Review-focused sentiment analysis using VADER and transformer models
- **Skill Extraction**: Semantic similarity-based skill extraction from review comments
- **Geographic Matching**: Proximity scoring using zipcode-based distance calculations
- **Dynamic Weighting**: User preference-based weight adjustment for personalized recommendations
- **Comprehensive Evaluation**: Built-in metrics for diversity, coverage, and relevance

### Algorithm Features
1. **Wilson Lower Bound**: Confidence-adjusted positive review scoring
2. **Bayesian Rating Shrinkage**: Handles agents with few reviews appropriately
3. **Exponential Decay**: Recency weighting for sales and reviews
4. **Multi-factor Scoring**: Combines 6+ major components with adjustable weights
5. **Negative Quality Penalties**: Detects and penalizes negative behaviors
6. **Buyer/Seller Fit Scoring**: Separate suitability scores based on review history

## 📁 Project Structure

```
agent_finder_backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models (request/response)
│   └── service.py           # Recommendation service logic
├── config/
│   └── settings.py          # Configuration settings
├── models/
│   ├── sentiment.py         # Sentiment analysis
│   ├── skills.py            # Skill extraction
│   └── scoring.py           # Agent scoring algorithms
├── utils/
│   ├── database.py          # Supabase client
│   ├── preprocessing.py     # Data preprocessing
│   ├── stats.py             # Statistical utilities
│   ├── geo.py               # Geographic utilities
│   └── evaluation.py        # Evaluation metrics
├── scripts/
│   └── process_data.py      # Data processing pipeline
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.9+
- Supabase account with database set up
- PostgreSQL database with schema as defined

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

### 4. Process Data

Run the data processing pipeline to engineer features:

```bash
python scripts/process_data.py
```

For a dry run (no database updates):
```bash
python scripts/process_data.py --dry-run
```

This will:
- Analyze sentiment for all reviews
- Extract skills from review comments
- Calculate aggregated metrics
- Apply statistical algorithms (Wilson LB, Bayesian shrinkage, etc.)
- Update database with processed features

### 5. Run the API

```bash
cd app
python main.py
```

Or with uvicorn:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8004
```

API will be available at `http://localhost:8004`

## 📚 API Documentation

### Interactive Docs
- Swagger UI: `http://localhost:8004/docs`
- ReDoc: `http://localhost:8004/redoc`

### Main Endpoints

#### 1. Search Agents
**POST** `/api/v1/agents/search`

Find and rank agents based on user criteria.

**Request Body:**
```json
{
  "user_type": "buyer",
  "state": "CA",
  "city": "Los Angeles",
  "min_price": 500000,
  "max_price": 1000000,
  "property_type": "single_family",
  "is_urgent": false,
  "language": "English",
  "sub_score_preferences": {
    "responsiveness": 0.4,
    "negotiation": 0.3,
    "professionalism": 0.2,
    "market_expertise": 0.1
  },
  "skill_preferences": {
    "communication": 0.5,
    "local_knowledge": 0.5
  },
  "additional_specializations": ["first_time_buyer"],
  "max_results": 10
}
```

**Response:**
```json
{
  "success": true,
  "message": "Found 10 matching agents",
  "total_results": 10,
  "recommendations": [
    {
      "advertiser_id": 123456,
      "full_name": "John Doe",
      "matching_score": 87.5,
      "proximity_score": 0.92,
      "distance_km": 5.3,
      "review_count": 50,
      "agent_rating": 4.8,
      "positive_review_count": 45,
      "negative_review_count": 2,
      "recently_sold_count": 20,
      "buyer_seller_fit": "both",
      ...
    }
  ],
  "search_params": {...}
}
```

#### 2. Get Agent Details
**GET** `/api/v1/agents/{agent_id}`

Get detailed information about a specific agent.

#### 3. Get Statistics
**GET** `/api/v1/stats`

Get system-wide statistics.

#### 4. Health Check
**GET** `/health`

Check API health status.

## 🧮 Algorithm Details

### Scoring Components

The matching score (0-100) is calculated from these weighted components:

1. **Buyer/Seller Fit (20%)**: How suitable the agent is for buyers or sellers
   - Based on historical review sentiment by role
   - Considers experience in each role

2. **Performance (25%)**: Recent sales activity and experience
   - Recency: Exponential decay based on days since last sale
   - Volume: Log-scaled recent sales count
   - Experience: Years of experience (normalized)

3. **Reviews (20%)**: Review quality and sentiment
   - Wilson Lower Bound for positive review confidence
   - Bayesian-shrunk ratings (handles low review counts)
   - Skills extracted from review comments (40% weight)

4. **Sub-scores (15%)**: User-weighted preferences
   - Responsiveness
   - Negotiation skills
   - Professionalism
   - Market expertise

5. **Proximity (10%)**: Geographic closeness
   - Exponential decay based on distance
   - Prioritizes base zipcode, then service areas

6. **Active Listings (10%)**: For urgent buyers/sellers only
   - Log-scaled count of active listings

### Statistical Methods

#### Wilson Lower Bound
```python
wilson_score = (p̂ + z²/2n ± z√(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)
```
Provides confidence-adjusted score for positive reviews.

#### Bayesian Rating Shrinkage
```python
shrunk_rating = (C × m + n × R) / (C + n)
```
Where:
- C = prior count (confidence)
- m = prior mean (global average)
- n = review count
- R = agent rating

#### Exponential Decay
```python
score = e^(-λt)
```
Applied to both sales recency and review age.

### Sentiment Analysis

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner):
- Tuned for social media and review text
- Fast and accurate for sentiment classification
- Fallback to rating-based inference when no comment

### Skill Extraction

Uses sentence transformers with semantic similarity:
1. Encode skill keywords and review comments
2. Calculate cosine similarity
3. Extract skills with similarity > threshold
4. Aggregate across all reviews with recency weighting

Positive skills: communication, local_knowledge, patience, honesty, etc.
Negative skills: unresponsive, pushy, unprofessional, etc.

## 📊 Evaluation

Run evaluation to assess recommendation quality:

```python
from utils.evaluation import run_evaluation_example

report, feature_importance = run_evaluation_example()
```

Metrics include:
- **Diversity**: Variety in recommendations
- **Coverage**: Breadth of agent pool covered
- **Relevance**: NDCG score (if ground truth available)
- **Feature Importance**: Correlation analysis

## 🔧 Configuration

Adjust algorithm parameters in `.env`:

```env
# Confidence for Wilson Lower Bound
WILSON_CONFIDENCE=0.95

# Bayesian prior for rating shrinkage
BAYESIAN_PRIOR_MEAN=4.0
BAYESIAN_PRIOR_COUNT=10

# Decay rates (daily)
RECENCY_DECAY_RATE=0.01
REVIEW_RECENCY_DECAY=0.005
DISTANCE_DECAY_RATE=0.02

# Base weights (auto-adjusted for urgent)
WEIGHT_BUYER_SELLER_FIT=0.20
WEIGHT_PERFORMANCE=0.25
WEIGHT_REVIEWS=0.20
WEIGHT_SUB_SCORES=0.15
WEIGHT_PROXIMITY=0.10
WEIGHT_ACTIVE_LISTINGS=0.10
```

## 🧪 Testing

Example search with curl:

```bash
curl -X POST "http://localhost:8000/api/v1/agents/search" \
  -H "Content-Type: application/json" \
  -d '{
    "user_type": "buyer",
    "state": "CA",
    "city": "Los Angeles",
    "min_price": 500000,
    "max_price": 1000000,
    "property_type": "single_family",
    "is_urgent": false,
    "max_results": 5
  }'
```

## 📈 Data Processing Pipeline

The `process_data.py` script performs:

1. **Load Data**: Fetch from Supabase
2. **Sentiment Analysis**: Analyze all review comments
3. **Skill Extraction**: Extract skills using embeddings
4. **Metric Aggregation**: Calculate per-agent statistics
5. **Feature Engineering**: 
   - Wilson scores
   - Bayesian shrinkage
   - Performance scores
   - Parse specializations
6. **Database Update**: Write processed features back

Run regularly (e.g., daily cron job) to keep features updated.

## 🔐 Database Schema

### Required Columns in `real_estate_agents`:
- Standard fields: advertiser_id, full_name, state, city, etc.
- Will be added by processing: positive_review_count, negative_review_count, wilson_score, shrunk_rating, performance_score, buyer_satisfaction, seller_satisfaction, avg_sub_*, skill_*, negative_*

### Required Columns in `reviews`:
- Standard fields: review_id, advertiser_id, review_rating, review_comment, etc.
- Will be added: sentiment, sentiment_confidence, days_since_review, recency_weight

## 💡 Usage Tips

1. **Run data processing first**: Always run `process_data.py` before starting the API
2. **Adjust weights**: Tune weights in `.env` based on your needs
3. **Monitor diversity**: Use evaluation metrics to ensure diverse recommendations
4. **Handle sparse data**: System gracefully handles missing reviews/scores
5. **Geographic coverage**: Expand search radius if too few results

## 🚧 Future Enhancements

- [ ] Collaborative filtering based on similar users
- [ ] Time-series analysis of agent performance
- [ ] A/B testing framework for algorithm improvements
- [ ] Caching layer for frequently searched locations
- [ ] Real-time data streaming for instant updates
- [ ] Advanced NLP for more nuanced skill extraction
- [ ] Graph-based agent relationships
- [ ] Automated hyperparameter tuning


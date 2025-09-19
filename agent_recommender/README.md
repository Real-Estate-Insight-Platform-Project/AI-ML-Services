# Top-K Agent Recommender System

A sophisticated real estate agent recommendation system that combines interpretable baseline scoring with advanced machine learning models to rank agents based on user preferences.

## 🎯 Overview

This system implements a **Top-K Agent Recommender** with two complementary approaches:

1. **Baseline Model**: Interpretable scoring using weighted features
2. **ML Model**: Advanced ranking with LightGBM/XGBoost and feature engineering
3. **Ensemble Model**: Combination of both approaches for optimal results

## 🏗️ Architecture

```
agent_recommender/
├── models/
│   ├── baseline_scorer.py      # Interpretable baseline scoring
│   └── ml_ranker.py           # ML-based ranking models
├── utils/
│   └── data_preprocessing.py   # Data loading and feature extraction
├── tests/
│   └── test_recommender.py    # Unit tests and evaluation
├── recommender.py             # Main system interface
├── demo.py                    # Demonstration script
└── requirements.txt           # Dependencies
```

## 🔬 Baseline Scoring Formula

The baseline model uses an interpretable weighted formula:

```
score(agent, query) = 
  0.30 × geo_overlap
+ 0.25 × price_band_match  
+ 0.15 × property_type_match
+ 0.15 × normalized_recency
+ 0.10 × rating_score
+ 0.03 × log(numReviews)
+ 0.02 × partner_premier_boost
```

### Key Features:

- **Geographic Fit**: Overlap between user's target regions and agent's service areas
- **Price Band Match**: Percentage of agent's deals within user's budget ±15%
- **Property Type Match**: Jaccard similarity between desired and handled property types
- **Recency & Volume**: Normalized recent activity and transaction volume
- **Reputation**: Star rating with Wilson score confidence intervals
- **Experience Signals**: Partner/Premier status, review count

## 🤖 ML Enhancement

The ML model extends the baseline with:

- **Advanced Feature Engineering**: Interaction features, experience scores, specialization indicators
- **LightGBM/XGBoost Rankers**: Learn complex patterns from data
- **Synthetic Training**: Uses baseline scores plus business logic for training labels
- **Feature Importance**: Explains which factors drive recommendations

## 🚀 Quick Start

### 1. Installation

```bash
cd agent_recommender
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from agent_recommender import AgentRecommenderSystem

# Initialize system
recommender = AgentRecommenderSystem("../agent_data/statewise_data")
recommender.initialize()

# Define user query
user_query = {
    'regions': ['Fort Myers', 'Naples'],
    'budget': 500000,
    'property_types': ['Single Family Residential', 'Condo/Co-op']
}

# Get recommendations
baseline_results = recommender.recommend(
    user_query, 
    model_type='baseline', 
    top_k=10, 
    explain=True
)

# ML recommendations (if model is trained)
ml_results = recommender.recommend(
    user_query, 
    model_type='ml', 
    top_k=10
)

# Ensemble approach
ensemble_results = recommender.recommend(
    user_query, 
    model_type='ensemble', 
    top_k=10
)
```

### 3. Run Demo

```bash
python demo.py
```

## 📊 Data Requirements

The system expects statewise JSON files with agent data:

```json
{
  "data": [
    {
      "agentId": 12345,
      "name": "John Doe",
      "starRating": 4.8,
      "numReviews": 45,
      "pastYearDeals": 25,
      "homeTransactionsLifetime": 150,
      "transactionVolumeLifetime": 75000000,
      "primaryServiceRegions": ["Downtown", "Midtown"],
      "propertyTypes": ["Single Family Residential", "Condo/Co-op"],
      "dealPrices": [400000, 550000, 350000, 600000],
      "businessMarket": "Miami",
      "partner": true,
      "isPremier": false,
      "brokerageName": "Redfin",
      "email": "john.doe@redfin.com",
      "phoneNumber": "(555) 123-4567"
    }
  ]
}
```

## 🔍 Features

### Baseline Scorer
- ✅ Interpretable weighted scoring
- ✅ Configurable feature weights
- ✅ Detailed score breakdown
- ✅ Business logic validation
- ✅ Geographic overlap calculation
- ✅ Price band matching
- ✅ Property type similarity
- ✅ Reputation scoring with confidence

### ML Ranker
- ✅ LightGBM and XGBoost support
- ✅ Advanced feature engineering
- ✅ Synthetic label generation
- ✅ Cross-validation
- ✅ Feature importance analysis
- ✅ Model persistence
- ✅ Ensemble methods

### System Features
- ✅ Unified API for all models
- ✅ Query validation and preprocessing
- ✅ Model comparison utilities
- ✅ Agent detail retrieval
- ✅ Performance monitoring
- ✅ Comprehensive testing
- ✅ Error handling and fallbacks

## 📈 Performance

The system is optimized for:
- **Fast Query Processing**: < 1 second for most queries
- **Scalability**: Handles 100K+ agents efficiently  
- **Memory Efficiency**: Lazy loading and feature caching
- **Reliability**: Automatic fallback to baseline if ML fails

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/test_recommender.py -v

# Run specific test
python tests/test_recommender.py TestBaselineScorer.test_baseline_score_calculation
```

## 📋 API Reference

### AgentRecommenderSystem

Main interface for the recommendation system.

#### Methods:

- `initialize()`: Load data and initialize models
- `recommend(query, model_type, top_k, explain)`: Get recommendations
- `compare_models(query, top_k)`: Compare different model outputs
- `get_agent_details(agent_id)`: Get detailed agent information
- `get_system_stats()`: Get system statistics

### Query Format

```python
user_query = {
    'regions': List[str],        # Target regions/cities
    'budget': float,             # Budget amount
    'property_types': List[str]  # Desired property types
}
```

### Response Format

```python
{
    'query': {...},              # Original query
    'recommendations': [...],    # Top-k agent recommendations
    'total_agents_evaluated': int,
    'model_type': str,
    'explanation': {...}         # Optional detailed explanation
}
```

## 🎛️ Configuration

### Baseline Weights

Customize feature weights in `BaselineAgentScorer`:

```python
custom_weights = {
    'geo_overlap': 0.35,
    'price_band_match': 0.30,
    'property_type_match': 0.20,
    'normalized_recency': 0.10,
    'rating_score': 0.05
}

scorer = BaselineAgentScorer(weights=custom_weights)
```

### ML Model Parameters

Configure ML models in `MLAgentRanker`:

```python
# LightGBM configuration
lgb_params = {
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8
}

ranker = MLAgentRanker('lightgbm')
# Modify ranker.model parameters before training
```

## 🔧 Troubleshooting

### Common Issues:

1. **Data Not Found**: Ensure the statewise data directory exists
2. **ML Model Training Fails**: Check data size and feature completeness
3. **Slow Performance**: Consider feature caching and data sampling
4. **Memory Issues**: Use batch processing for large datasets

### Debug Mode:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed processing information
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is part of the AI-ML Services platform for real estate analytics.

## 🔮 Future Enhancements

- [ ] Deep learning models (neural networks)
- [ ] Real-time model updates
- [ ] A/B testing framework
- [ ] Multi-objective optimization
- [ ] Geographic clustering
- [ ] Natural language query processing
- [ ] Integration with external APIs

---

**Built with ❤️ for real estate professionals**
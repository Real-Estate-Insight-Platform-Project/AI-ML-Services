# testing/functional_testing/agent_recommender/conftest.py - Shared fixtures for agent recommender tests
import pytest
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient
from fastapi.testclient import TestClient
import pandas as pd

# Add agent-finder-approach-1 to Python path
agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'agent-finder-approach-1'))
sys.path.insert(0, agent_path)

# Set test environment variables
os.environ.update({
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_SERVICE_KEY": "test-key-12345",
    "CORS_ORIGINS": "*"
})

# Mock supabase client creation before importing main
mock_supabase_client = MagicMock()
mock_query = MagicMock()
mock_query.select.return_value = mock_query
mock_query.eq.return_value = mock_query
mock_query.gte.return_value = mock_query
mock_query.neq.return_value = mock_query
mock_query.overlaps.return_value = mock_query
mock_query.order.return_value = mock_query
mock_query.range.return_value = mock_query
mock_supabase_client.table.return_value = mock_query

# Patch the create_client function before importing main
with patch('supabase.create_client', return_value=mock_supabase_client):
    try:
        import importlib.util
        import importlib
        
        # Import main module from the agent-finder-approach-1 directory
        main_path = os.path.join(agent_path, 'main.py')
        spec = importlib.util.spec_from_file_location("main", main_path)
        main = importlib.util.module_from_spec(spec)
        sys.modules['main'] = main
        spec.loader.exec_module(main)
    except ImportError as e:
        print(f"Import error: {e}")
        print(f"Python path: {sys.path}")
        print(f"Agent path: {agent_path}")
        raise

@pytest.fixture
def mock_supabase():
    """Mock Supabase client for testing."""
    mock_client = MagicMock()
    
    # Mock successful table operations - create complete chain
    mock_query = MagicMock()
    # All query methods return the same mock_query for chaining
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.gte.return_value = mock_query
    mock_query.lte.return_value = mock_query  # Add this method that was missing
    mock_query.neq.return_value = mock_query
    mock_query.overlaps.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.range.return_value = mock_query
    
    # Mock the execute response with proper structure
    mock_response = MagicMock()
    mock_response.count = 2
    mock_response.data = []  # Will be overridden in test_client fixture
    mock_response.error = None
    mock_query.execute.return_value = mock_response
    
    mock_client.table.return_value = mock_query
    
    return mock_client

@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing."""
    return [
        {
            "agent_id": 1,
            "name": "John Smith",
            "brokerage_name": "Premium Realty",
            "star_rating": 4.8,
            "num_reviews": 125,
            "past_year_deals": 45,
            "avg_transaction_value": 650000.0,
            "primary_service_regions": ["Miami, Florida", "Fort Lauderdale, Florida"],
            "comprehensive_service_areas": ["Miami, Florida", "Fort Lauderdale, Florida", "Broward County"],
            "business_market_normalized": "Miami-Fort Lauderdale-West Palm Beach",
            "office_state": "Florida",
            "property_types": ["Single Family", "Condo"],
            "phone_number": "+1-555-0123",
            "email": "john.smith@premiumrealty.com",
            "profile_url": "https://example.com/agent/1",
            "photo_url": "https://example.com/photos/1.jpg",
            "is_premier": True,
            "is_active": True,
            "deal_prices_median": 550000.0,
            "deal_prices_min": 200000.0,
            "deal_prices_max": 1200000.0,
            "deal_prices_std": 85000.0,
            "num_comprehensive_areas": 3,
            "partner": False,
            "serves_offers": True,
            "serves_listings": True,
            "profile_contact_enabled": True,
            "business_market_id": 1,
            "home_transactions_lifetime": 280,
            "transaction_volume_lifetime": 45000000.0
        },
        {
            "agent_id": 2,
            "name": "Sarah Johnson",
            "brokerage_name": "Elite Properties",
            "star_rating": 4.6,
            "num_reviews": 89,
            "past_year_deals": 32,
            "avg_transaction_value": 480000.0,
            "primary_service_regions": ["Orlando, Florida"],
            "comprehensive_service_areas": ["Orlando, Florida", "Orange County"],
            "business_market_normalized": "Orlando-Kissimmee-Sanford",
            "office_state": "Florida",
            "property_types": ["Single Family", "Townhouse"],
            "phone_number": "+1-555-0456",
            "email": "sarah.johnson@eliteprops.com",
            "profile_url": "https://example.com/agent/2",
            "photo_url": "https://example.com/photos/2.jpg",
            "is_premier": False,
            "is_active": True,
            "deal_prices_median": 420000.0,
            "deal_prices_min": 150000.0,
            "deal_prices_max": 800000.0,
            "deal_prices_std": 65000.0,
            "num_comprehensive_areas": 2,
            "partner": True,
            "serves_offers": True,
            "serves_listings": True,
            "profile_contact_enabled": True,
            "business_market_id": 2,
            "home_transactions_lifetime": 180,
            "transaction_volume_lifetime": 28000000.0
        }
    ]

@pytest.fixture
def mock_scorer():
    """Mock AgentScorer for testing."""
    mock_scorer = MagicMock()
    
    # Mock scoring method that returns structured breakdown
    def mock_total(df, locations, property_types, price_range):
        n_agents = len(df)
        
        # Create mock breakdown DataFrame
        breakdown_data = {
            'performance_score': [0.85, 0.75][:n_agents],
            'expertise_score': [0.90, 0.80][:n_agents],
            'satisfaction_score': [0.95, 0.85][:n_agents],
            'professional_score': [0.88, 0.78][:n_agents],
            'availability_score': [0.92, 0.82][:n_agents],
            'total_score': [4.50, 4.00][:n_agents]
        }
        
        breakdown_df = pd.DataFrame(breakdown_data)
        total_scores = breakdown_df['total_score']
        
        return total_scores, breakdown_df
    
    mock_scorer.total.side_effect = mock_total
    return mock_scorer

@pytest.fixture
async def test_client(mock_supabase, mock_scorer, sample_agent_data):
    """Create test client with mocked dependencies."""
    
    with patch.object(main, 'supabase', mock_supabase), \
         patch.object(main, 'scorer', mock_scorer):
        
        # Configure the mock query response with sample data
        mock_supabase.table().select().execute().data = sample_agent_data
        mock_supabase.table().select().execute().count = len(sample_agent_data)
        mock_supabase.table().select().execute().error = None
        
        async with AsyncClient(app=main.app, base_url="http://test") as client:
            yield client

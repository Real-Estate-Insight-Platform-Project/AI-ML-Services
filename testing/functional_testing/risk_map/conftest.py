# testing/risk_map/conftest.py - Shared fixtures for risk-map tests
import pytest
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient

# Add risk-map to Python path
risk_map_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'risk-map'))
sys.path.insert(0, risk_map_path)

# Set test environment variables
os.environ["MAP_DATABASE_URL"] = "postgresql://test:test@localhost:5432/test_db"

@pytest.fixture
def mock_asyncpg_pool():
    """Mock asyncpg connection pool for testing."""
    from unittest.mock import MagicMock
    
    mock_conn = AsyncMock()
    
    # Create proper async context manager for acquire()
    class MockAsyncContextManager:
        def __init__(self, connection):
            self.connection = connection
            
        async def __aenter__(self):
            return self.connection
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None
    
    # Create a proper mock pool
    mock_pool = MagicMock()
    
    # Make acquire() return the context manager directly (not a callable)
    mock_pool.acquire.return_value = MockAsyncContextManager(mock_conn)
    
    return mock_pool, mock_conn

@pytest.fixture
def sample_mvt_data():
    """Sample MVT (Mapbox Vector Tile) binary data."""
    # This is a minimal valid MVT binary sequence
    return b'\x1a\x12\x08\x01\x12\x08\x18\x02\x22\x02\x00\x01\x28\x01'

@pytest.fixture
def sample_property_data():
    """Sample property data for testing."""
    return {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "title": "Beautiful Waterfront Home",
        "address": "123 Ocean Drive",
        "city": "Miami Beach",
        "state": "FL",
        "bedrooms": 3,
        "bathrooms": 2.5,
        "square_feet": 2500,
        "price": 850000.00
    }

@pytest.fixture
async def test_client(mock_asyncpg_pool, sample_mvt_data, sample_property_data):
    """Create test client with mocked dependencies."""
    mock_pool, mock_conn = mock_asyncpg_pool
    
    # Mock database responses
    async def mock_fetchrow(sql, *args):
        if "us_counties_mvt" in sql:
            return {"mvt": sample_mvt_data}
        elif "properties_mvt" in sql:
            return {"mvt": sample_mvt_data}
        elif "properties" in sql and "WHERE id" in sql:
            # Check if the provided ID looks like a valid UUID
            property_id = args[0] if args else ""
            # Simple UUID format check (8-4-4-4-12 hex characters)
            import re
            uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            if re.match(uuid_pattern, str(property_id).lower()):
                return sample_property_data
            else:
                return None
        return None
    
    mock_conn.fetchrow.side_effect = mock_fetchrow
    
    with patch('asyncpg.create_pool', return_value=mock_pool):
        try:
            import importlib.util
            import importlib
            
            # Import app.main module from the risk-map directory
            app_main_path = os.path.join(risk_map_path, 'app', 'main.py')
            spec = importlib.util.spec_from_file_location("app.main", app_main_path)
            app_main = importlib.util.module_from_spec(spec)
            sys.modules['app.main'] = app_main
            spec.loader.exec_module(app_main)
            app = app_main.app
        except ImportError as e:
            print(f"Import error: {e}")
            print(f"Risk map path: {risk_map_path}")
            raise
        
        # Manually set the pool on app state for testing
        app.state.pool = mock_pool
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
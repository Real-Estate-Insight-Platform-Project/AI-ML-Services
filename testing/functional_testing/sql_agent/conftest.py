# testing/sql_agent/conftest.py - Shared fixtures for sql-agent tests
import pytest
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Add sql-agent to Python path
sql_agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sql-agent'))
sys.path.insert(0, sql_agent_path)

# Set test environment variables
os.environ.update({
    "GOOGLE_API_KEY": "test-google-api-key",
    "SQL_AGENT_DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
    "LOG_SQL": "false",
    "GEMINI_MODEL": "gemini-1.5-flash"
})

@pytest.fixture
def mock_sql_database():
    """Mock SQLDatabase for testing."""
    mock_db = MagicMock()
    
    # Mock table info
    mock_db.get_table_info.return_value = """
    Table: nri_counties
    Columns: state_name (text), county_name (text), risk_index_score (numeric)
    
    Table: predictions
    Columns: year (integer), month (integer), state (text), median_listing_price (numeric), market_trend (text)
    
    Table: properties
    Columns: id (uuid), title (text), address (text), city (text), state (text), price (numeric), bedrooms (integer), bathrooms (numeric), square_feet (integer), created_at (timestamp)
    
    Table: state_market
    Columns: year (integer), month (integer), state (text), median_price (numeric), listing_count (integer)
    """
    
    # Mock successful query execution
    def mock_run(sql, *args, **kwargs):
        if "SELECT" not in sql.upper():
            raise ValueError("Only SELECT queries are permitted.")
        
        # Return different mock results based on query type
        if "properties" in sql.lower() and "title" in sql.lower():
            return "Title|Address|City|State|Bedrooms|Bathrooms|Price\nBeautiful Home|123 Main St|Miami|FL|3|2|500000\nModern Condo|456 Ocean Dr|Miami Beach|FL|2|2|650000"
        elif "predictions" in sql.lower():
            return "state|median_listing_price|market_trend\nFlorida|450000|rising\nCalifornia|800000|stable"
        elif "COUNT" in sql.upper():
            return "count\n150"
        else:
            return "state|value\nFlorida|500000\nCalifornia|750000"
    
    mock_db.run.side_effect = mock_run
    return mock_db

@pytest.fixture
def mock_llm():
    """Mock ChatGoogleGenerativeAI for testing."""
    mock_llm = MagicMock()
    return mock_llm

@pytest.fixture
def mock_agent():
    """Mock SQL agent for testing."""
    mock_agent = MagicMock()
    
    def mock_run(question):
        # Check for SQL injection patterns FIRST (before other patterns)
        if ("injection" in question.lower() or "drop table" in question.lower() or 
              "delete from" in question.lower() or "update " in question.lower() or 
              "insert into" in question.lower()):
            raise ValueError("Only SELECT queries are permitted.")
        elif "error" in question.lower():
            raise Exception("Database connection failed")
        elif "properties" in question.lower() and "miami" in question.lower():
            return """| Title | Address | City | State | Bedrooms | Bathrooms | Price |
|-------|---------|------|-------|----------|-----------|-------|
| Beautiful Home | 123 Main St | Miami | FL | 3 | 2 | 500000 |
| Modern Condo | 456 Ocean Dr | Miami Beach | FL | 2 | 2 | 650000 |"""
        elif ("states" in question.lower() or "state" in question.lower()) and "price" in question.lower():
            return """| State | Average Listing Price |
|-------|----------------------|
| Florida | $450,000 |
| California | $800,000 |"""
        else:
            return "Based on the data, here are the results showing relevant information."
    
    mock_agent.run.side_effect = mock_run
    return mock_agent

@pytest.fixture
async def test_client(mock_sql_database, mock_llm, mock_agent):
    """Create test client with mocked dependencies."""
    
    try:
        import importlib.util
        import importlib
        
        # Patch ALL the imports BEFORE importing main to avoid validation issues
        with patch('langchain_community.agent_toolkits.sql.base.create_sql_agent', return_value=mock_agent):
            with patch('langchain_community.utilities.SQLDatabase', return_value=mock_sql_database):
                with patch('langchain_google_genai.ChatGoogleGenerativeAI', return_value=mock_llm):
                    with patch('langchain_community.agent_toolkits.SQLDatabaseToolkit') as mock_toolkit:
                        with patch('sqlalchemy.create_engine') as mock_engine:
                            with patch('sqlalchemy.event.listens_for') as mock_event:
                                # Mock the toolkit constructor to avoid validation
                                mock_toolkit.return_value = MagicMock()
                                mock_engine.return_value = MagicMock()
                                mock_event.return_value = lambda func: func  # Mock decorator
                                
                                # Import main module from the sql-agent directory
                                main_path = os.path.join(sql_agent_path, 'main.py')
                                spec = importlib.util.spec_from_file_location("main", main_path)
                                main = importlib.util.module_from_spec(spec)
                                sys.modules['main'] = main
                                spec.loader.exec_module(main)
        
        app = main.app
    except ImportError as e:
        print(f"Import error: {e}")
        print(f"SQL agent path: {sql_agent_path}")
        raise
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
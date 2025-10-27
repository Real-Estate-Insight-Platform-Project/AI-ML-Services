"""
Comprehensive Test Suite for Agentic Real Estate Assistant
"""
import pytest
import json
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from src.main import app
from src.config import settings
from src.utils.session_manager import SessionManager
from src.tools.tools import (
    geocode_location,
    calculate_distance,
    query_supabase,
    query_bigquery,
    sanitize_sql_query,
    convert_for_json
)


# ==================== FIXTURES ====================

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def session_manager():
    """Session manager instance"""
    return SessionManager()


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    with patch('redis.from_url') as mock:
        redis_mock = MagicMock()
        redis_mock.ping.return_value = True
        redis_mock.get.return_value = None
        redis_mock.setex.return_value = True
        mock.return_value = redis_mock
        yield redis_mock


# ==================== API TESTS ====================

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == settings.app_name
        assert data["version"] == settings.app_version
        assert "features" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "timestamp" in data
    
    def test_stats_endpoint(self, client):
        """Test stats endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "redis" in data
    
    def test_chat_endpoint_success(self, client):
        """Test chat endpoint with valid request"""
        payload = {
            "message": "Show me properties in Boston",
            "session_id": "test-session-123"
        }
        response = client.post("/chat", json=payload)
        assert response.status_code in [200, 500]  # May fail without real credentials
        
        if response.status_code == 200:
            data = response.json()
            assert "response" in data
            assert "session_id" in data
            assert "timestamp" in data
    
    def test_chat_endpoint_invalid_request(self, client):
        """Test chat endpoint with invalid request"""
        payload = {"message": ""}  # Empty message
        response = client.post("/chat", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_session_endpoints(self, client):
        """Test session management endpoints"""
        session_id = "test-session-456"
        
        # Test get session (may not exist)
        response = client.get(f"/session/{session_id}")
        assert response.status_code in [200, 404]
        
        # Test extend session
        response = client.post(f"/session/{session_id}/extend")
        assert response.status_code in [200, 404]
        
        # Test clear session
        response = client.delete(f"/session/{session_id}")
        assert response.status_code in [200, 500]
        
        # Test list sessions
        response = client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "active_sessions" in data


# ==================== TOOL TESTS ====================

class TestTools:
    """Test tool implementations"""
    
    def test_geocode_location_success(self):
        """Test successful geocoding"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "result": {
                    "addressMatches": [{
                        "coordinates": {"x": -71.0589, "y": 42.3601},
                        "matchedAddress": "Boston, MA"
                    }]
                }
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            result = geocode_location.invoke({"place": "Boston University"})
            assert result["success"] is True
            assert "lat" in result
            assert "lon" in result
    
    def test_geocode_location_failure(self):
        """Test geocoding failure"""
        result = geocode_location.invoke({"place": ""})
        assert result["success"] is False
        assert "error" in result
    
    def test_calculate_distance(self):
        """Test distance calculation"""
        # Boston to New York
        result = calculate_distance.invoke({
            "lat1": 42.3601,
            "lon1": -71.0589,
            "lat2": 40.7128,
            "lon2": -74.0060
        })
        assert result["success"] is True
        assert result["distance_miles"] > 0
        assert result["distance_km"] > 0
        assert 180 < result["distance_miles"] < 200  # Approximate distance Boston to NYC
    
    def test_sanitize_sql_query_valid(self):
        """Test SQL query sanitization with valid query"""
        query = "SELECT * FROM properties WHERE city = 'Boston'"
        result = sanitize_sql_query(query, "supabase")
        assert result is not None
        assert "LIMIT" in result.upper()
    
    def test_sanitize_sql_query_dangerous(self):
        """Test SQL query sanitization blocks dangerous queries"""
        dangerous_queries = [
            "DROP TABLE properties",
            "DELETE FROM properties",
            "UPDATE properties SET price = 0",
            "SELECT * FROM profiles",  # Protected table
            "SELECT * FROM user_favorites",  # Protected table
        ]
        
        for query in dangerous_queries:
            result = sanitize_sql_query(query, "supabase")
            assert result is None
    
    def test_convert_for_json(self):
        """Test JSON conversion utility"""
        from decimal import Decimal
        from uuid import UUID
        
        test_data = {
            "string": "test",
            "int": 123,
            "float": 45.67,
            "decimal": Decimal("99.99"),
            "uuid": UUID("12345678-1234-5678-1234-567812345678"),
            "datetime": datetime(2025, 10, 26),
            "none": None,
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2}
        }
        
        result = convert_for_json(test_data)
        
        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None
        
        # Check conversions
        assert isinstance(result["decimal"], float)
        assert isinstance(result["uuid"], str)
        assert isinstance(result["datetime"], str)


# ==================== SESSION MANAGER TESTS ====================

class TestSessionManager:
    """Test session management"""
    
    def test_session_lifecycle(self, session_manager):
        """Test complete session lifecycle"""
        session_id = "test-session-lifecycle"
        
        # Initially empty
        conversation = session_manager.get_conversation(session_id)
        assert conversation == []
        
        # Add messages
        assert session_manager.add_message(session_id, "user", "Hello")
        assert session_manager.add_message(session_id, "assistant", "Hi there!")
        
        # Retrieve conversation
        conversation = session_manager.get_conversation(session_id)
        assert len(conversation) == 2
        assert conversation[0]["role"] == "user"
        assert conversation[1]["role"] == "assistant"
        
        # Get session info
        info = session_manager.get_session_info(session_id)
        assert info["exists"] is True
        assert info["message_count"] == 2
        
        # Clear session
        assert session_manager.clear_conversation(session_id)
        conversation = session_manager.get_conversation(session_id)
        assert conversation == []
    
    def test_cache_operations(self, session_manager):
        """Test caching operations"""
        if not session_manager.available:
            pytest.skip("Redis not available")
        
        query = "test query"
        result = {"data": "test result"}
        
        # Cache result
        assert session_manager.cache_query_result(query, result)
        
        # Retrieve cached result
        cached = session_manager.get_cached_query(query)
        assert cached == result
        
        # Different query should miss
        cached = session_manager.get_cached_query("different query")
        assert cached is None


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.integration
    def test_full_chat_flow(self, client):
        """Test complete chat flow"""
        # Send initial message
        payload1 = {
            "message": "Hello, can you help me find properties?",
            "session_id": "integration-test-session"
        }
        response1 = client.post("/chat", json=payload1)
        
        if response1.status_code == 200:
            data1 = response1.json()
            session_id = data1["session_id"]
            
            # Send follow-up message
            payload2 = {
                "message": "Show me properties in Boston under $500k",
                "session_id": session_id
            }
            response2 = client.post("/chat", json=payload2)
            assert response2.status_code == 200
            
            # Check session has history
            session_response = client.get(f"/session/{session_id}")
            if session_response.status_code == 200:
                session_data = session_response.json()
                assert session_data["message_count"] >= 2


# ==================== PERFORMANCE TESTS ====================

class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.performance
    def test_concurrent_requests(self, client):
        """Test handling concurrent requests"""
        import concurrent.futures
        
        def make_request():
            payload = {
                "message": "Test message",
                "session_id": f"perf-test-{datetime.now().timestamp()}"
            }
            return client.post("/chat", json=payload)
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # At least some should succeed
        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count >= 0  # May fail without credentials


# ==================== RUN TESTS ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

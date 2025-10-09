# tests/integration/test_sql_agent_api.py
import pytest
from httpx import AsyncClient
import json
from unittest.mock import patch

class TestHealthEndpoint:
    """Test suite for health/root endpoint."""
    
    @pytest.mark.asyncio
    async def test_root_endpoint_success(self, test_client: AsyncClient):
        """Test root endpoint returns service information."""
        response = await test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert data["message"] == "SQL Agent API is running."

class TestAskEndpointBasicFunctionality:
    """Test basic functionality of the /ask endpoint."""
    
    @pytest.mark.asyncio
    async def test_ask_simple_question_success(self, test_client: AsyncClient):
        """Test asking a simple question returns structured response."""
        payload = {"question": "What states have rising price trends?"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
    
    @pytest.mark.asyncio
    async def test_ask_property_search_question(self, test_client: AsyncClient):
        """Test property search question with formatting."""
        payload = {"question": "Show me properties in Miami"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        # Should format properties in card style
        assert "Here are the top matching properties" in data["answer"]
    
    @pytest.mark.asyncio
    async def test_ask_state_aggregation_question(self, test_client: AsyncClient):
        """Test state aggregation question with formatting."""
        payload = {"question": "What are the average listing prices by state?"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        # Should format state values nicely
        assert "Here are the results:" in data["answer"]

class TestAskEndpointInputValidation:
    """Test input validation and error handling."""
    
    @pytest.mark.asyncio
    async def test_ask_empty_question(self, test_client: AsyncClient):
        """Test asking empty question."""
        payload = {"question": ""}
        
        response = await test_client.post("/ask", json=payload)
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]
    
    @pytest.mark.asyncio
    async def test_ask_missing_question_field(self, test_client: AsyncClient):
        """Test request missing question field."""
        payload = {}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    async def test_ask_invalid_json_payload(self, test_client: AsyncClient):
        """Test invalid JSON payload."""
        response = await test_client.post(
            "/ask",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_ask_wrong_content_type(self, test_client: AsyncClient):
        """Test wrong content type."""
        response = await test_client.post(
            "/ask",
            data="question=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == 422

class TestAskEndpointBoundaryValues:
    """Test boundary values for question inputs."""
    
    @pytest.mark.asyncio
    async def test_ask_very_long_question(self, test_client: AsyncClient):
        """Test very long question (boundary value)."""
        long_question = "What are the " + "best " * 500 + "properties in Miami?"
        payload = {"question": long_question}
        
        response = await test_client.post("/ask", json=payload)
        
        # Should handle long questions gracefully
        assert response.status_code in [200, 400, 413]  # 413 = Payload Too Large
    
    @pytest.mark.asyncio
    async def test_ask_single_character_question(self, test_client: AsyncClient):
        """Test single character question (boundary value)."""
        payload = {"question": "?"}
        
        response = await test_client.post("/ask", json=payload)
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    @pytest.mark.asyncio
    async def test_ask_unicode_question(self, test_client: AsyncClient):
        """Test question with Unicode characters."""
        payload = {"question": "Show properties in Miami 🏠 with prices > $500k 💰"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

class TestAskEndpointSQLInjectionProtection:
    """Test SQL injection protection mechanisms."""
    
    @pytest.mark.asyncio
    async def test_ask_sql_injection_attempt(self, test_client: AsyncClient):
        """Test SQL injection attempt is blocked."""
        payload = {"question": "'; DROP TABLE properties; --"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return error message about forbidden operations
        assert "error" in data or "Only SELECT queries are permitted" in data.get("answer", "")
    
    @pytest.mark.asyncio
    async def test_ask_delete_attempt(self, test_client: AsyncClient):
        """Test DELETE statement attempt is blocked."""
        payload = {"question": "DELETE FROM properties WHERE price > 1000000"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should be blocked by SQL safety measures
        assert "error" in data or "Only SELECT" in data.get("answer", "")
    
    @pytest.mark.asyncio
    async def test_ask_update_attempt(self, test_client: AsyncClient):
        """Test UPDATE statement attempt is blocked."""
        payload = {"question": "UPDATE properties SET price = 0 WHERE city = 'Miami'"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200 
        data = response.json()
        
        # Should be blocked by SQL safety measures
        assert "error" in data or "Only SELECT" in data.get("answer", "")
    
    @pytest.mark.asyncio
    async def test_ask_insert_attempt(self, test_client: AsyncClient):
        """Test INSERT statement attempt is blocked."""
        payload = {"question": "INSERT INTO properties (title, price) VALUES ('Test', 100000)"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should be blocked by SQL safety measures
        assert "error" in data or "Only SELECT" in data.get("answer", "")

class TestAskEndpointEquivalencePartitioning:
    """Test equivalence partitioning for different question types."""
    
    @pytest.mark.asyncio
    async def test_ask_property_search_queries(self, test_client: AsyncClient):
        """Test property search query equivalence class."""
        property_questions = [
            "Show me properties in Miami",
            "Find houses in Orlando",
            "List properties under $500k in Tampa",
            "What properties are available in Jacksonville?"
        ]
        
        for question in property_questions:
            payload = {"question": question}
            response = await test_client.post("/ask", json=payload)
            
            assert response.status_code == 200, f"Failed for question: {question}"
            data = response.json()
            assert "answer" in data
    
    @pytest.mark.asyncio
    async def test_ask_state_analysis_queries(self, test_client: AsyncClient):
        """Test state analysis query equivalence class."""
        state_questions = [
            "Which states have the highest property prices?",
            "Show me price trends by state",
            "What are average listing prices per state?",
            "Compare property values across states"
        ]
        
        for question in state_questions:
            payload = {"question": question}
            response = await test_client.post("/ask", json=payload)
            
            assert response.status_code == 200, f"Failed for question: {question}"
            data = response.json()
            assert "answer" in data
    
    @pytest.mark.asyncio
    async def test_ask_prediction_queries(self, test_client: AsyncClient):
        """Test prediction query equivalence class."""
        prediction_questions = [
            "What are the price predictions for next month?",
            "Show me market trends for 2024",
            "Which states are predicted to have rising prices?",
            "What is the forecast for Florida real estate?"
        ]
        
        for question in prediction_questions:
            payload = {"question": question}
            response = await test_client.post("/ask", json=payload)
            
            assert response.status_code == 200, f"Failed for question: {question}"
            data = response.json()
            assert "answer" in data
    
    @pytest.mark.asyncio
    async def test_ask_risk_analysis_queries(self, test_client: AsyncClient):
        """Test risk analysis query equivalence class."""
        risk_questions = [
            "What are the risk scores by county?",
            "Show me low-risk areas for investment",
            "Which counties have the highest risk index?",
            "Compare risk levels across states"
        ]
        
        for question in risk_questions:
            payload = {"question": question}
            response = await test_client.post("/ask", json=payload)
            
            assert response.status_code == 200, f"Failed for question: {question}"
            data = response.json()
            assert "answer" in data

class TestAskEndpointResponseFormatting:
    """Test response formatting functionality."""
    
    @pytest.mark.asyncio
    async def test_ask_property_card_formatting(self, test_client: AsyncClient):
        """Test property results are formatted as cards."""
        payload = {"question": "Show me properties in Miami"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        answer = data["answer"]
        
        # Should format as numbered property cards
        assert "Here are the top matching properties" in answer
        assert "1)" in answer  # Numbered list
        # Should contain property details without IDs
        assert "Beautiful Home" in answer or "properties" in answer.lower()
    
    @pytest.mark.asyncio
    async def test_ask_state_value_formatting(self, test_client: AsyncClient):
        """Test state values are formatted nicely."""
        payload = {"question": "What are the average listing prices by state?"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        answer = data["answer"]
        
        # Should format state values nicely
        assert "Here are the results:" in answer
        # Should contain formatted state data
        assert "Florida" in answer or "state" in answer.lower()
    
    @pytest.mark.asyncio
    async def test_ask_no_formatting_fallback(self, test_client: AsyncClient):
        """Test fallback when no special formatting applies."""
        payload = {"question": "How many total properties are in the database?"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return raw answer without special formatting
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

class TestAskEndpointErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_ask_database_connection_error(self, test_client: AsyncClient):
        """Test handling of database connection errors."""
        payload = {"question": "This should cause an error"}
        
        # Mock agent to raise exception
        with patch('main.agent') as mock_agent:
            mock_agent.run.side_effect = Exception("Database connection failed")
            
            response = await test_client.post("/ask", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should return error in response
            assert "error" in data
            assert "Database connection failed" in data["error"]
    
    @pytest.mark.asyncio
    async def test_ask_sql_execution_error(self, test_client: AsyncClient):
        """Test handling of SQL execution errors."""
        payload = {"question": "This should cause SQL injection error"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should handle SQL safety violations gracefully
        if "error" in data:
            assert isinstance(data["error"], str)
        else:
            assert "answer" in data

class TestAskEndpointTableAccessControl:
    """Test table access control and allowlist functionality."""
    
    @pytest.mark.asyncio
    async def test_ask_allowed_tables_access(self, test_client: AsyncClient):
        """Test access to allowed tables works."""
        allowed_table_questions = [
            "Show data from nri_counties table",
            "Query the predictions table",
            "Find properties in the properties table",
            "Show state_market data"
        ]
        
        for question in allowed_table_questions:
            payload = {"question": question}
            response = await test_client.post("/ask", json=payload)
            
            assert response.status_code == 200, f"Failed for question: {question}"
            data = response.json()
            assert "answer" in data or "error" in data  # Should process the question
    
    @pytest.mark.asyncio
    async def test_ask_forbidden_table_references(self, test_client: AsyncClient):
        """Test references to forbidden tables are handled."""
        forbidden_questions = [
            "Show me user profiles",
            "Access the user_favorites table",
            "Query user account information",
            "Show personal data from profiles"
        ]
        
        for question in forbidden_questions:
            payload = {"question": question}
            response = await test_client.post("/ask", json=payload)
            
            assert response.status_code == 200, f"Failed for question: {question}"
            data = response.json()
            
            # Should either handle gracefully or return error
            assert "answer" in data or "error" in data

class TestAskEndpointLimitEnforcement:
    """Test LIMIT clause enforcement and row limiting."""
    
    @pytest.mark.asyncio
    async def test_ask_automatic_limit_addition(self, test_client: AsyncClient):
        """Test automatic LIMIT clause addition for multi-row queries."""
        payload = {"question": "Show me all properties in Florida"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        if "Here are the top matching properties" in data.get("answer", ""):
            # Should mention showing up to 10 results
            assert "showing up to 10" in data["answer"]
    
    @pytest.mark.asyncio
    async def test_ask_respect_existing_limits(self, test_client: AsyncClient):
        """Test existing LIMIT clauses are respected."""
        payload = {"question": "Show me 5 properties with highest prices"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    @pytest.mark.asyncio
    async def test_ask_no_limit_for_aggregates(self, test_client: AsyncClient):
        """Test no LIMIT enforcement for single aggregate results."""
        payload = {"question": "What is the average property price?"}
        
        response = await test_client.post("/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

class TestAskEndpointPerformance:
    """Test performance-related aspects."""
    
    @pytest.mark.asyncio
    async def test_ask_response_time_reasonable(self, test_client: AsyncClient):
        """Test response time is reasonable."""
        import time
        
        payload = {"question": "Show me properties in Miami"}
        
        start_time = time.time()
        response = await test_client.post("/ask", json=payload)
        end_time = time.time()
        
        assert response.status_code == 200
        # Should respond within reasonable time (adjust threshold as needed)
        assert (end_time - start_time) < 10.0  # 10 second threshold for LLM processing
    
    @pytest.mark.asyncio
    async def test_ask_concurrent_requests(self, test_client: AsyncClient):
        """Test handling of concurrent requests."""
        import asyncio
        
        async def make_request():
            payload = {"question": "What are property prices in Florida?"}
            return await test_client.post("/ask", json=payload)
        
        # Make 3 concurrent requests
        tasks = [make_request() for _ in range(3)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data or "error" in data
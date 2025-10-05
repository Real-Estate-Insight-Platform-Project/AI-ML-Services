# tests/integration/test_agent_recommender_api.py
import pytest
from httpx import AsyncClient
import json

class TestHealthEndpoint:
    """Test suite for health check endpoint."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, test_client: AsyncClient):
        """Test health endpoint returns success."""
        response = await test_client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

class TestLocationsEndpoint:
    """Test suite for locations endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_locations_success(self, test_client: AsyncClient):
        """Test locations endpoint returns standardized locations."""
        response = await test_client.get("/locations")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify structure
        assert "major_markets" in data
        assert "florida_markets" in data
        assert "states" in data
        
        # Verify content types
        assert isinstance(data["major_markets"], list)
        assert isinstance(data["florida_markets"], list)
        assert isinstance(data["states"], list)
        
        # Verify expected locations present
        assert "Miami, Florida" in data["major_markets"]
        assert "Miami, Florida" in data["florida_markets"]
        assert "Florida" in data["states"]

class TestDebugEndpoint:
    """Test suite for debug endpoint."""
    
    @pytest.mark.asyncio
    async def test_debug_count_success(self, test_client: AsyncClient):
        """Test debug count endpoint returns agent count."""
        response = await test_client.get("/debug/count")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "count" in data
        assert "error" in data
        assert isinstance(data["count"], int)

class TestRecommendEndpoint:
    """Test suite for agent recommendation endpoint using equivalence partitioning and boundary value analysis."""
    
    @pytest.mark.asyncio
    async def test_recommend_minimal_valid_request(self, test_client: AsyncClient):
        """Test minimal valid recommendation request."""
        payload = {"top_k": 5}
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_matches" in data
        assert "returned" in data
        assert "results" in data
        
        assert isinstance(data["total_matches"], int)
        assert isinstance(data["returned"], int)
        assert isinstance(data["results"], list)
        
        # Verify agent structure if results exist
        if data["results"]:
            agent = data["results"][0]
            required_fields = [
                "rank", "agent_id", "name", "star_rating", "num_reviews", 
                "past_year_deals", "avg_transaction_value", "service_regions",
                "comprehensive_areas", "property_types", "is_premier", 
                "is_active", "total_score", "breakdown"
            ]
            
            for field in required_fields:
                assert field in agent, f"Missing field: {field}"
    
    @pytest.mark.asyncio
    async def test_recommend_with_locations_valid(self, test_client: AsyncClient):
        """Test recommendation with valid location filters."""
        payload = {
            "locations": ["Miami, Florida", "Orlando, Florida"],
            "top_k": 10
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["returned"] <= 10
    
    @pytest.mark.asyncio
    async def test_recommend_with_property_types_valid(self, test_client: AsyncClient):
        """Test recommendation with valid property type filters."""
        payload = {
            "property_types": ["Single Family", "Condo"],
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["returned"] <= 5
    
    @pytest.mark.asyncio
    async def test_recommend_with_price_range_valid(self, test_client: AsyncClient):
        """Test recommendation with valid price range (equivalence partitioning: valid range)."""
        payload = {
            "price_min": 200000.0,
            "price_max": 800000.0,
            "top_k": 8
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["returned"] <= 8
    
    @pytest.mark.asyncio
    async def test_recommend_with_rating_and_reviews_valid(self, test_client: AsyncClient):
        """Test recommendation with rating and review filters."""
        payload = {
            "min_rating": 4.0,
            "min_reviews": 50,
            "top_k": 3
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["returned"] <= 3
    
    @pytest.mark.asyncio
    async def test_recommend_with_phone_requirement(self, test_client: AsyncClient):
        """Test recommendation with phone requirement."""
        payload = {
            "require_phone": True,
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify agents with phone numbers
        for agent in data["results"]:
            if agent["phone"]:
                assert agent["phone"] != "Not Available"
    
    @pytest.mark.asyncio
    async def test_recommend_comprehensive_filters(self, test_client: AsyncClient):
        """Test recommendation with all filters combined."""
        payload = {
            "locations": ["Miami, Florida"],
            "property_types": ["Single Family"],
            "price_min": 300000.0,
            "price_max": 700000.0,
            "top_k": 3,
            "min_rating": 4.5,
            "min_reviews": 80,
            "require_phone": True
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["returned"] <= 3

class TestRecommendEndpointBoundaryValues:
    """Test boundary values for recommendation endpoint."""
    
    @pytest.mark.asyncio
    async def test_recommend_top_k_boundary_minimum(self, test_client: AsyncClient):
        """Test top_k minimum boundary value (1)."""
        payload = {"top_k": 1}
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["returned"] <= 1
    
    @pytest.mark.asyncio
    async def test_recommend_top_k_boundary_large(self, test_client: AsyncClient):
        """Test top_k large boundary value (100)."""
        payload = {"top_k": 100}
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["returned"] <= 100
    
    @pytest.mark.asyncio
    async def test_recommend_price_boundary_zero_min(self, test_client: AsyncClient):
        """Test price_min boundary value (0)."""
        payload = {
            "price_min": 0.0,
            "price_max": 500000.0,
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_recommend_rating_boundary_zero(self, test_client: AsyncClient):
        """Test min_rating boundary value (0.0)."""
        payload = {
            "min_rating": 0.0,
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_recommend_rating_boundary_five(self, test_client: AsyncClient):
        """Test min_rating boundary value (5.0)."""
        payload = {
            "min_rating": 5.0,
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_recommend_reviews_boundary_zero(self, test_client: AsyncClient):
        """Test min_reviews boundary value (0)."""
        payload = {
            "min_reviews": 0,
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200

class TestRecommendEndpointErrorCases:
    """Test error cases for recommendation endpoint."""
    
    @pytest.mark.asyncio
    async def test_recommend_invalid_top_k_zero(self, test_client: AsyncClient):
        """Test invalid top_k value (0) - should fail validation."""
        payload = {"top_k": 0}
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_recommend_invalid_top_k_negative(self, test_client: AsyncClient):
        """Test invalid top_k value (negative) - should fail validation."""
        payload = {"top_k": -5}
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_recommend_invalid_price_negative_min(self, test_client: AsyncClient):
        """Test invalid price_min (negative) - should fail validation."""
        payload = {
            "price_min": -100000.0,
            "price_max": 500000.0,
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_recommend_invalid_price_negative_max(self, test_client: AsyncClient):
        """Test invalid price_max (negative) - should fail validation."""
        payload = {
            "price_min": 200000.0,
            "price_max": -500000.0,
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_recommend_partial_price_range_min_only(self, test_client: AsyncClient):
        """Test providing only price_min without price_max - should fail."""
        payload = {
            "price_min": 200000.0,
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert "both price_min and price_max" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_recommend_partial_price_range_max_only(self, test_client: AsyncClient):
        """Test providing only price_max without price_min - should fail."""
        payload = {
            "price_max": 800000.0,
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert "both price_min and price_max" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_recommend_invalid_json(self, test_client: AsyncClient):
        """Test invalid JSON payload."""
        response = await test_client.post(
            "/recommend", 
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

class TestRecommendEndpointLocationNormalization:
    """Test location normalization functionality."""
    
    @pytest.mark.asyncio
    async def test_recommend_location_case_insensitive(self, test_client: AsyncClient):
        """Test location matching is case insensitive."""
        payload = {
            "locations": ["miami, florida", "ORLANDO, FLORIDA"],
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_recommend_location_state_abbreviation(self, test_client: AsyncClient):
        """Test state abbreviation normalization."""
        payload = {
            "locations": ["FL", "ca"],
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_recommend_location_partial_match(self, test_client: AsyncClient):
        """Test partial location matching."""
        payload = {
            "locations": ["Miami", "Orlando"],
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_recommend_location_empty_list(self, test_client: AsyncClient):
        """Test empty locations list."""
        payload = {
            "locations": [],
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_recommend_location_whitespace_handling(self, test_client: AsyncClient):
        """Test location whitespace handling."""
        payload = {
            "locations": ["  Miami, Florida  ", "\tOrlando, Florida\n"],
            "top_k": 5
        }
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200

class TestRecommendEndpointResponseFormat:
    """Test response format and data validation."""
    
    @pytest.mark.asyncio
    async def test_recommend_response_structure_complete(self, test_client: AsyncClient):
        """Test complete response structure validation."""
        payload = {"top_k": 2}
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify top-level structure
        assert set(data.keys()) == {"total_matches", "returned", "results"}
        
        if data["results"]:
            agent = data["results"][0]
            
            # Verify all agent fields
            expected_fields = {
                "rank", "agent_id", "name", "brokerage", "star_rating", 
                "num_reviews", "past_year_deals", "avg_transaction_value",
                "service_regions", "comprehensive_areas", "business_market",
                "office_state", "property_types", "phone", "email",
                "profile_url", "photo_url", "is_premier", "is_active",
                "total_score", "breakdown"
            }
            
            assert set(agent.keys()) == expected_fields
            
            # Verify breakdown structure
            breakdown = agent["breakdown"]
            expected_breakdown_fields = {
                "performance", "market_expertise", "client_satisfaction",
                "professional_standing", "availability"
            }
            
            assert set(breakdown.keys()) == expected_breakdown_fields
            
            # Verify data types
            assert isinstance(agent["rank"], int)
            assert isinstance(agent["agent_id"], int)
            assert isinstance(agent["name"], str)
            assert isinstance(agent["star_rating"], (int, float))
            assert isinstance(agent["num_reviews"], int)
            assert isinstance(agent["service_regions"], list)
            assert isinstance(agent["comprehensive_areas"], list)
            assert isinstance(agent["property_types"], list)
            assert isinstance(agent["is_premier"], bool)
            assert isinstance(agent["is_active"], bool)
            assert isinstance(agent["total_score"], (int, float))
    
    @pytest.mark.asyncio
    async def test_recommend_response_ranking_order(self, test_client: AsyncClient):
        """Test agents are returned in correct ranking order."""
        payload = {"top_k": 5}
        
        response = await test_client.post("/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        if len(data["results"]) > 1:
            # Verify ranking order
            for i, agent in enumerate(data["results"], 1):
                assert agent["rank"] == i
            
            # Verify score ordering (descending)
            scores = [agent["total_score"] for agent in data["results"]]
            assert scores == sorted(scores, reverse=True)
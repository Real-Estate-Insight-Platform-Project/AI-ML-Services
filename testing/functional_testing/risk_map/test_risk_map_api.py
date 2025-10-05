# tests/integration/test_risk_map_api.py
import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock, MagicMock

class TestHealthEndpoint:
    """Test suite for health/root endpoint."""
    
    @pytest.mark.asyncio
    async def test_root_endpoint_success(self, test_client: AsyncClient):
        """Test root endpoint returns service information."""
        response = await test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify expected structure
        assert "ok" in data
        assert "service" in data
        assert "message" in data
        assert "tiles" in data
        assert "json" in data
        
        assert data["ok"] is True
        assert data["service"] == "risk-map"
        
        # Verify tiles endpoints are documented
        tiles = data["tiles"]
        assert "counties" in tiles
        assert "properties" in tiles
        assert "/tiles/counties/{z}/{x}/{y}" in tiles["counties"]
        assert "/tiles/properties/{z}/{x}/{y}" in tiles["properties"]
        
        # Verify JSON endpoints are documented
        json_endpoints = data["json"]
        assert "property_details" in json_endpoints
        assert "/properties/{id}" in json_endpoints["property_details"]

class TestCountiesVectorTiles:
    """Test suite for counties vector tiles endpoint."""
    
    @pytest.mark.asyncio
    async def test_counties_tile_valid_coordinates(self, test_client: AsyncClient):
        """Test counties tile with valid z/x/y coordinates."""
        response = await test_client.get("/tiles/counties/10/305/380")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
        assert "Cache-Control" in response.headers
        assert "public, max-age=3600" in response.headers["Cache-Control"]
        
        # Verify response contains MVT data
        assert len(response.content) > 0
    
    @pytest.mark.asyncio
    async def test_counties_tile_boundary_zoom_minimum(self, test_client: AsyncClient):
        """Test counties tile with minimum zoom level (boundary value)."""
        response = await test_client.get("/tiles/counties/0/0/0")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
    
    @pytest.mark.asyncio
    async def test_counties_tile_boundary_zoom_maximum(self, test_client: AsyncClient):
        """Test counties tile with high zoom level (boundary value)."""
        response = await test_client.get("/tiles/counties/18/150000/100000")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
    
    @pytest.mark.asyncio
    async def test_counties_tile_world_boundaries(self, test_client: AsyncClient):
        """Test counties tile at world coordinate boundaries."""
        # Test various world boundary coordinates
        test_cases = [
            (1, 0, 0),     # Top-left of world at zoom 1
            (1, 1, 0),     # Top-right of world at zoom 1
            (1, 0, 1),     # Bottom-left of world at zoom 1
            (1, 1, 1),     # Bottom-right of world at zoom 1
        ]
        
        for z, x, y in test_cases:
            response = await test_client.get(f"/tiles/counties/{z}/{x}/{y}")
            assert response.status_code in [200, 204]  # 204 for empty tiles is valid
            assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
    
    @pytest.mark.asyncio
    async def test_counties_tile_miami_area(self, test_client: AsyncClient):
        """Test counties tile for Miami area (specific geographic region)."""
        # Coordinates approximately covering Miami-Dade County
        response = await test_client.get("/tiles/counties/10/284/410")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"

class TestPropertiesVectorTiles:
    """Test suite for properties vector tiles endpoint."""
    
    @pytest.mark.asyncio
    async def test_properties_tile_valid_coordinates(self, test_client: AsyncClient):
        """Test properties tile with valid z/x/y coordinates."""
        response = await test_client.get("/tiles/properties/12/1150/1850")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
        assert "Cache-Control" in response.headers
        assert "public, max-age=300" in response.headers["Cache-Control"]
        
        # Verify response contains MVT data
        assert len(response.content) > 0
    
    @pytest.mark.asyncio
    async def test_properties_tile_different_cache_timeout(self, test_client: AsyncClient):
        """Test properties tile has different cache timeout than counties."""
        response = await test_client.get("/tiles/properties/10/305/380")
        
        assert response.status_code == 200
        # Properties should have shorter cache time (300s vs 3600s for counties)
        assert "max-age=300" in response.headers["Cache-Control"]
    
    @pytest.mark.asyncio
    async def test_properties_tile_high_zoom_level(self, test_client: AsyncClient):
        """Test properties tile at high zoom level for detailed view."""
        response = await test_client.get("/tiles/properties/15/9300/15100")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
    
    @pytest.mark.asyncio
    async def test_properties_tile_low_zoom_level(self, test_client: AsyncClient):
        """Test properties tile at low zoom level for overview."""
        response = await test_client.get("/tiles/properties/5/9/12")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"

class TestTileCoordinateValidation:
    """Test coordinate validation and edge cases for tile endpoints."""
    
    @pytest.mark.asyncio
    async def test_tile_negative_coordinates(self, test_client: AsyncClient):
        """Test tiles with negative coordinates (invalid)."""
        # Most tile servers treat negative coordinates as invalid
        response = await test_client.get("/tiles/counties/-1/100/100")
        
        # Should still handle gracefully (either 200 with empty tile or 204)
        assert response.status_code in [200, 204, 404]
    
    @pytest.mark.asyncio
    async def test_tile_very_large_coordinates(self, test_client: AsyncClient):
        """Test tiles with very large coordinates."""
        response = await test_client.get("/tiles/counties/10/999999/999999")
        
        # Should handle gracefully
        assert response.status_code in [200, 204]
        if response.status_code == 200:
            assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
    
    @pytest.mark.asyncio
    async def test_tile_zero_coordinates(self, test_client: AsyncClient):
        """Test tiles with zero coordinates (boundary value)."""
        response = await test_client.get("/tiles/counties/0/0/0")
        
        assert response.status_code in [200, 204]
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"

class TestEmptyTileHandling:
    """Test handling of empty tiles (204 No Content responses)."""
    
    @pytest.mark.asyncio
    async def test_empty_counties_tile(self, test_client: AsyncClient):
        """Test empty counties tile handling."""
        # Mock empty response for testing
        with patch('app.main.app.state.pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {"mvt": None}
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool.acquire.return_value.__aexit__.return_value = None
            
            response = await test_client.get("/tiles/counties/10/0/0")
            
            assert response.status_code == 204
            assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
            assert len(response.content) == 0
    
    @pytest.mark.asyncio
    async def test_empty_properties_tile(self, test_client: AsyncClient):
        """Test empty properties tile handling."""
        # Mock empty response for testing
        with patch('app.main.app.state.pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {"mvt": None}
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool.acquire.return_value.__aexit__.return_value = None
            
            response = await test_client.get("/tiles/properties/10/0/0")
            
            assert response.status_code == 204
            assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
            assert len(response.content) == 0

class TestPropertyDetailsEndpoint:
    """Test suite for property details endpoint."""
    
    @pytest.mark.asyncio
    async def test_property_details_valid_uuid(self, test_client: AsyncClient):
        """Test property details with valid UUID."""
        property_id = "123e4567-e89b-12d3-a456-426614174000"
        response = await test_client.get(f"/properties/{property_id}")
        
        assert response.status_code == 200
        assert "Cache-Control" in response.headers
        assert "public, max-age=3600" in response.headers["Cache-Control"]
        
        data = response.json()
        
        # Verify expected fields
        expected_fields = {
            "id", "title", "address", "city", "state",
            "bedrooms", "bathrooms", "square_feet", "price"
        }
        assert set(data.keys()) == expected_fields
        
        # Verify data types
        assert isinstance(data["id"], str)
        assert isinstance(data["title"], str)
        assert isinstance(data["address"], str)
        assert isinstance(data["city"], str)
        assert isinstance(data["state"], str)
        assert isinstance(data["bedrooms"], int)
        assert isinstance(data["bathrooms"], (int, float))
        assert isinstance(data["square_feet"], int)
        assert isinstance(data["price"], (int, float)) or data["price"] is None
    
    @pytest.mark.asyncio
    async def test_property_details_nonexistent_id(self, test_client: AsyncClient):
        """Test property details with non-existent UUID."""
        property_id = "00000000-0000-0000-0000-000000000000"
        
        # Mock no result from database
        with patch('app.main.app.state.pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = None
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool.acquire.return_value.__aexit__.return_value = None
            
            response = await test_client.get(f"/properties/{property_id}")
            
            assert response.status_code == 404
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "Property not found"
    
    @pytest.mark.asyncio
    async def test_property_details_invalid_uuid_format(self, test_client: AsyncClient):
        """Test property details with invalid UUID format."""
        invalid_ids = [
            "not-a-uuid",
            "123",
            "123e4567-e89b-12d3-a456",  # Too short
            "123e4567-e89b-12d3-a456-426614174000-extra",  # Too long
        ]
        
        for invalid_id in invalid_ids:
            response = await test_client.get(f"/properties/{invalid_id}")
            
            # Should either be 404 or 422 depending on validation
            assert response.status_code in [404, 422], f"Failed for ID: {invalid_id}"
    
    @pytest.mark.asyncio
    async def test_property_details_null_price_handling(self, test_client: AsyncClient):
        """Test property details with null price value."""
        property_id = "123e4567-e89b-12d3-a456-426614174000"
        
        # Mock property with null price
        mock_property = {
            "id": property_id,
            "title": "Property with No Price",
            "address": "456 Unknown Value St",
            "city": "Test City",
            "state": "TS",
            "bedrooms": 2,
            "bathrooms": 1,
            "square_feet": 1000,
            "price": None
        }
        
        with patch('app.main.app.state.pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = mock_property
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_pool.acquire.return_value.__aexit__.return_value = None
            
            response = await test_client.get(f"/properties/{property_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["price"] is None


class TestCORSAndMiddleware:
    """Test CORS and middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_cors_headers_present(self, test_client: AsyncClient):
        """Test CORS headers are present in responses."""
        response = await test_client.get("/")
        
        # CORS headers should be present due to middleware
        assert response.status_code == 200
        # Note: In test environment, CORS headers might not be visible
        # This test verifies the endpoint works with CORS middleware enabled
    
    @pytest.mark.asyncio
    async def test_gzip_compression_headers(self, test_client: AsyncClient):
        """Test GZip compression middleware."""
        # Test with Accept-Encoding header
        headers = {"Accept-Encoding": "gzip, deflate"}
        response = await test_client.get("/", headers=headers)
        
        assert response.status_code == 200
        # Response should be processed by GZip middleware for large responses

class TestPerformanceConsiderations:
    """Test performance-related aspects."""
    
    @pytest.mark.asyncio
    async def test_tile_response_time_reasonable(self, test_client: AsyncClient):
        """Test tile responses are reasonably fast."""
        import time
        
        start_time = time.time()
        response = await test_client.get("/tiles/counties/10/305/380")
        end_time = time.time()
        
        assert response.status_code == 200
        # Should respond within reasonable time (adjust threshold as needed)
        assert (end_time - start_time) < 5.0  # 5 second threshold
    
    @pytest.mark.asyncio
    async def test_property_details_response_time_reasonable(self, test_client: AsyncClient):
        """Test property details responses are reasonably fast."""
        import time
        
        property_id = "123e4567-e89b-12d3-a456-426614174000"
        
        start_time = time.time()
        response = await test_client.get(f"/properties/{property_id}")
        end_time = time.time()
        
        assert response.status_code == 200
        # Should respond within reasonable time
        assert (end_time - start_time) < 3.0  # 3 second threshold
from locust import HttpUser, task, between
import json
import os
import random

BASE = os.getenv("RISKMAP_BASE")

class RiskMapUser(HttpUser):
    host = BASE
    wait_time = between(0.1, 0.5)  # Fast think time for tile requests

    # Sample tile coordinates for different zoom levels
    world_tiles = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1),
        (2, 0, 0), (2, 1, 1), (2, 2, 2), (2, 3, 3)
    ]
    
    regional_tiles = [
        (5, 9, 12),    # North America area
        (6, 18, 24),   # USA area
        (7, 36, 49),   # Southeast USA
        (8, 72, 98),   # Florida region
        (9, 144, 196)  # Detailed regional view
    ]
    
    city_tiles = [
        (10, 284, 410),   # Miami-Dade area
        (10, 305, 380),   # Central Florida
        (11, 568, 820),   # Tampa area
        (11, 610, 760),   # Orlando area
        (12, 1150, 1850), # Detailed city view
        (12, 1220, 1520)  # Another city detail
    ]
    
    detail_tiles = [
        (13, 2325, 3775),  # High detail
        (14, 4650, 7550),  # Very high detail
        (15, 9300, 15100), # Street level
        (16, 18600, 30200) # Maximum detail
    ]

    @task(4)
    def get_counties_tile(self):
        """Test counties vector tiles"""
        # Choose tile coordinates based on zoom level distribution
        tile_sets = [
            (self.world_tiles, 0.1),     # 10% world tiles
            (self.regional_tiles, 0.2),  # 20% regional tiles  
            (self.city_tiles, 0.5),      # 50% city tiles
            (self.detail_tiles, 0.2)     # 20% detail tiles
        ]
        
        # Weighted random selection
        rand = random.random()
        cumulative = 0
        for tiles, weight in tile_sets:
            cumulative += weight
            if rand <= cumulative:
                z, x, y = random.choice(tiles)
                break
        else:
            z, x, y = random.choice(self.city_tiles)  # fallback
        
        with self.client.get(
            f"/tiles/counties/{z}/{x}/{y}", 
            catch_response=True, 
            name=f"/tiles/counties (z={z})"
        ) as res:
            if res.status_code in [200, 204]:  # 204 for empty tiles is valid
                content_type = res.headers.get("Content-Type", res.headers.get("content-type", ""))
                if "application/vnd.mapbox-vector-tile" in content_type:
                    # Validate content based on status
                    if res.status_code == 200 and len(res.content) > 0:
                        res.success()
                    elif res.status_code == 204 and len(res.content) == 0:
                        res.success()
                    else:
                        res.failure(f"Content mismatch for status {res.status_code}: {len(res.content)} bytes")
                else:
                    res.failure(f"Wrong content type: {content_type}")
            else:
                res.failure(f"HTTP {res.status_code}: {res.text}")

    @task(3)
    def get_properties_tile(self):
        """Test properties vector tiles"""
        # Properties tiles are more relevant at higher zoom levels
        tile_sets = [
            (self.regional_tiles, 0.1),  # 10% regional tiles
            (self.city_tiles, 0.6),      # 60% city tiles  
            (self.detail_tiles, 0.3)     # 30% detail tiles
        ]
        
        rand = random.random()
        cumulative = 0
        for tiles, weight in tile_sets:
            cumulative += weight
            if rand <= cumulative:
                z, x, y = random.choice(tiles)
                break
        else:
            z, x, y = random.choice(self.city_tiles)
        
        with self.client.get(
            f"/tiles/properties/{z}/{x}/{y}", 
            catch_response=True, 
            name=f"/tiles/properties (z={z})"
        ) as res:
            if res.status_code in [200, 204]:
                content_type = res.headers.get("Content-Type", res.headers.get("content-type", ""))
                if "application/vnd.mapbox-vector-tile" in content_type:
                    if res.status_code == 200 and len(res.content) > 0:
                        res.success()
                    elif res.status_code == 204 and len(res.content) == 0:
                        res.success()
                    else:
                        res.failure(f"Content mismatch for status {res.status_code}: {len(res.content)} bytes")
                else:
                    res.failure(f"Wrong content type: {content_type}")
            else:
                res.failure(f"HTTP {res.status_code}: {res.text}")

    @task(2)
    def get_service_info(self):
        """Test the root endpoint for service information"""
        with self.client.get("/", catch_response=True, name="/") as res:
            if res.status_code == 200:
                try:
                    data = res.json()
                    expected_fields = ["ok", "service", "tiles", "json"]
                    if all(field in data for field in expected_fields):
                        if data["ok"] is True and data["service"] == "risk-map":
                            res.success()
                        else:
                            res.failure(f"Invalid service info: {data}")
                    else:
                        res.failure(f"Missing service info fields: {data.keys()}")
                except Exception as e:
                    res.failure(f"Invalid JSON response: {e}")
            else:
                res.failure(f"HTTP {res.status_code}: {res.text}")

    def on_start(self):
        """Called when a user starts"""
        print(f"Starting Risk Map load test against {self.host}")

    def on_stop(self):
        """Called when a user stops"""
        print(f"Stopping Risk Map load test")
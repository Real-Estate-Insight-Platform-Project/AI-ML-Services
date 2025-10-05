from locust import HttpUser, task, between
import json
import os
import random

BASE = os.getenv("RECOMMENDER_BASE")

class AgentRecommenderUser(HttpUser):
    host = BASE
    wait_time = between(2, 4)  # longer wait time for lighter load

    # Sample test data for realistic requests
    locations_options = [
        ["Miami, Florida"],
        ["Orlando, Florida"], 
        ["Tampa, Florida"],
        ["Miami, Florida", "Orlando, Florida"],
        ["Jacksonville, Florida"],
        [],  # empty for testing all locations
    ]
    
    property_types_options = [
        ["Single Family"],
        ["Condo"],
        ["Single Family", "Condo"],
        ["Single Family", "Condo", "Townhouse"],
        [],  # empty for all property types
    ]
    
    price_ranges = [
        (200000, 500000),
        (300000, 700000),
        (400000, 800000),
        (500000, 1000000),
        (None, None)  # No price filter
    ]

    @task(3)
    def recommend_agents(self):
        """Main recommendation endpoint test"""
        # Create realistic request payload
        price_min, price_max = random.choice(self.price_ranges)
        
        payload = {
            "top_k": random.choice([5, 10, 15, 20]),
            "locations": random.choice(self.locations_options),
            "property_types": random.choice(self.property_types_options),
            "min_rating": random.choice([None, 4.0, 4.5]),
            "min_reviews": random.choice([None, 50, 100]),
            "require_phone": random.choice([True, False])
        }
        
        # Add price filters if specified
        if price_min is not None and price_max is not None:
            payload["price_min"] = price_min
            payload["price_max"] = price_max
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        with self.client.post(
            "/recommend", 
            json=payload, 
            headers={"Content-Type": "application/json"},
            catch_response=True, 
            name="/recommend"
        ) as res:
            if res.status_code == 200:
                try:
                    data = res.json()
                    # Validate response structure
                    required_fields = ["total_matches", "returned", "results"]
                    if all(field in data for field in required_fields):
                        # Validate that returned count matches results length
                        if data["returned"] == len(data["results"]):
                            # Validate agent structure if results exist
                            if data["results"]:
                                agent = data["results"][0]
                                agent_fields = ["rank", "agent_id", "name", "star_rating", "total_score"]
                                if all(field in agent for field in agent_fields):
                                    res.success()
                                else:
                                    res.failure(f"Missing agent fields: {agent.keys()}")
                            else:
                                res.success()  # Valid response with no results
                        else:
                            res.failure(f"Returned count mismatch: {data['returned']} vs {len(data['results'])}")
                    else:
                        res.failure(f"Missing required fields: {data.keys()}")
                except Exception as e:
                    res.failure(f"Invalid JSON response: {e}")
            else:
                res.failure(f"HTTP {res.status_code}: {res.text}")

    @task(1)
    def get_locations(self):
        """Test the locations endpoint"""
        with self.client.get("/locations", catch_response=True, name="/locations") as res:
            if res.status_code == 200:
                try:
                    data = res.json()
                    if isinstance(data, dict) and "major_markets" in data:
                        res.success()
                    else:
                        res.failure(f"Unexpected locations format: {type(data)}")
                except Exception as e:
                    res.failure(f"Invalid JSON response: {e}")
            else:
                res.failure(f"HTTP {res.status_code}: {res.text}")

    @task(2)
    def health_check(self):
        """Test the health endpoint"""
        with self.client.get("/health", catch_response=True, name="/health") as res:
            if res.status_code == 200:
                try:
                    data = res.json()
                    if "status" in data and data["status"] == "ok":
                        res.success()
                    else:
                        res.failure(f"Unexpected health response: {data}")
                except Exception as e:
                    res.failure(f"Invalid JSON response: {e}")
            else:
                res.failure(f"Health check failed: HTTP {res.status_code}")

    @task(1)
    def debug_count(self):
        """Test the debug count endpoint"""
        with self.client.get("/debug/count", catch_response=True, name="/debug/count") as res:
            if res.status_code == 200:
                try:
                    data = res.json()
                    if "count" in data and isinstance(data["count"], int):
                        res.success()
                    else:
                        res.failure(f"Unexpected debug response: {data}")
                except Exception as e:
                    res.failure(f"Invalid JSON response: {e}")
            else:
                res.failure(f"HTTP {res.status_code}: {res.text}")

    def on_start(self):
        """Called when a user starts"""
        print(f"Starting Agent Recommender load test against {self.host}")

    def on_stop(self):
        """Called when a user stops"""
        print(f"Stopping Agent Recommender load test")
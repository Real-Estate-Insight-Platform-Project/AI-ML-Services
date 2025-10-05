from locust import HttpUser, task, between
import json
import os
import random

BASE = os.getenv("SQL_AGENT_BASE")
API_KEY = os.getenv("SQL_AGENT_API_KEY", "")

# Load prompts from the datasets directory
dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "sql_agent_prompts.json")
with open(dataset_path, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

class SQLAgentUser(HttpUser):
    host = BASE
    wait_time = between(5, 10)  # much longer wait time for AI queries

    @task
    def ask_question(self):
        q = random.choice(PROMPTS)
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        
        with self.client.post(
            "/ask",  # Correct endpoint based on main.py
            json={"question": q}, 
            headers=headers, 
            catch_response=True, 
            name="/ask"
        ) as res:
            if res.status_code == 200:
                try:
                    data = res.json()
                    if "answer" in data or "error" in data:
                        res.success()
                    else:
                        res.failure(f"Missing expected fields in response: {data}")
                except Exception as e:
                    res.failure(f"Invalid JSON response: {e}")
            else:
                res.failure(f"HTTP {res.status_code}: {res.text}")

    @task(2)  # Run this task twice as often
    def health_check(self):
        """Simple health check endpoint to verify service availability"""
        with self.client.get("/", catch_response=True, name="/") as res:  # Root endpoint returns service info
            if res.status_code == 200:
                try:
                    data = res.json()
                    if "message" in data:
                        res.success()
                    else:
                        res.failure(f"Unexpected response format: {data}")
                except Exception as e:
                    res.failure(f"Invalid JSON response: {e}")
            else:
                res.failure(f"Health check failed: HTTP {res.status_code}")

    def on_start(self):
        """Called when a user starts - can be used for setup"""
        print(f"Starting user session against {self.host}")

    def on_stop(self):
        """Called when a user stops - can be used for cleanup"""
        print(f"Stopping user session")
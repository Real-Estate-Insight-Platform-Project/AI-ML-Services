#!/usr/bin/env python3
"""
Test script for Agent Recommender API

This script demonstrates how to interact with the standalone Agent Recommender FastAPI service.
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8001"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service is {data['status']}")
            if data.get('healthy'):
                print(f"   📊 Total agents: {data.get('total_agents', 'N/A')}")
                print(f"   🗺️  States covered: {data.get('unique_states', 'N/A')}")
                print(f"   🏢 Markets: {data.get('unique_markets', 'N/A')}")
        else:
            print(f"⚠️ Health check failed with status {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to agent recommender service")
        print("   Make sure the service is running on http://localhost:8001")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    return True

def test_recommendations():
    """Test the recommendation endpoint."""
    print("\n🎯 Testing agent recommendations...")
    
    test_query = {
        "regions": ["Miami", "Fort Lauderdale"],
        "budget": 500000,
        "property_types": ["Single Family Residential", "Condo"],
        "top_k": 3,
        "model_type": "baseline",
        "explain": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/recommend", json=test_query)
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get('recommendations', [])
            print(f"✅ Got {len(recommendations)} recommendations")
            
            for i, rec in enumerate(recommendations[:2], 1):  # Show first 2
                print(f"\n   🏆 Agent #{i}:")
                print(f"      Name: {rec.get('name', 'N/A')}")
                print(f"      Rating: {rec.get('starRating', 'N/A')} ⭐")
                print(f"      Reviews: {rec.get('numReviews', 'N/A')}")
                print(f"      Score: {rec.get('score', 'N/A'):.3f}")
                print(f"      Market: {rec.get('businessMarket', 'N/A')}")
            
            if data.get('explanation'):
                print(f"\n   💡 Model used: {data.get('model_type', 'N/A')}")
                
        else:
            print(f"⚠️ Recommendation request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Recommendation test error: {e}")

def test_agent_details():
    """Test the agent details endpoint."""
    print("\n👤 Testing agent details...")
    
    # First get a recommendation to get a valid agent ID
    test_query = {
        "regions": ["Tampa"],
        "budget": 300000,
        "property_types": ["Single Family Residential"],
        "top_k": 1,
        "model_type": "baseline"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/recommend", json=test_query)
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get('recommendations', [])
            
            if recommendations:
                agent_id = recommendations[0].get('agentId')
                print(f"   🔍 Looking up agent ID: {agent_id}")
                
                # Get detailed info
                detail_response = requests.get(f"{BASE_URL}/agents/{agent_id}")
                if detail_response.status_code == 200:
                    agent = detail_response.json()
                    print(f"✅ Agent details retrieved:")
                    print(f"      Name: {agent.get('name', 'N/A')}")
                    print(f"      Email: {agent.get('email', 'N/A')}")
                    print(f"      Phone: {agent.get('phoneNumber', 'N/A')}")
                    print(f"      Brokerage: {agent.get('brokerageName', 'N/A')}")
                    print(f"      Lifetime deals: {agent.get('homeTransactionsLifetime', 'N/A')}")
                else:
                    print(f"⚠️ Agent details request failed: {detail_response.status_code}")
            else:
                print("⚠️ No agents found to test details with")
        else:
            print(f"⚠️ Could not get agent for details test: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Agent details test error: {e}")

def test_system_stats():
    """Test the system stats endpoint."""
    print("\n📊 Testing system statistics...")
    
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ System statistics:")
            print(f"      Total agents: {stats.get('total_agents', 'N/A'):,}")
            print(f"      States covered: {stats.get('unique_states', 'N/A')}")
            print(f"      Markets: {stats.get('unique_markets', 'N/A')}")
            print(f"      Brokerages: {stats.get('unique_brokerages', 'N/A')}")
            
            data_summary = stats.get('data_summary', {})
            print(f"      Avg rating: {data_summary.get('avg_star_rating', 0):.2f}")
            print(f"      Avg reviews: {data_summary.get('avg_reviews', 0):.1f}")
            
            models = stats.get('models_available', {})
            print(f"      Models: Baseline: {models.get('baseline', False)}, ML: {models.get('ml', False)}")
        else:
            print(f"⚠️ Stats request failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ System stats test error: {e}")

def test_model_comparison():
    """Test the model comparison endpoint."""
    print("\n🆚 Testing model comparison...")
    
    test_query = {
        "regions": ["Orlando"],
        "budget": 400000,
        "property_types": ["Single Family Residential"],
        "top_k": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/compare", json=test_query)
        if response.status_code == 200:
            data = response.json()
            models_compared = data.get('models_compared', [])
            print(f"✅ Compared {len(models_compared)} models: {', '.join(models_compared)}")
            
            if 'overlap_analysis' in data:
                overlap = data['overlap_analysis']
                print(f"      Agents in common: {overlap.get('agents_in_common', 0)}")
                print(f"      Overlap percentage: {overlap.get('overlap_percentage', 0):.1f}%")
            
        else:
            print(f"⚠️ Model comparison failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Model comparison test error: {e}")

def main():
    """Run all tests."""
    print("🚀 Testing Agent Recommender API")
    print("=" * 50)
    
    # Check if service is running
    if not test_health_check():
        print("\n❌ Service not available. Please start the agent recommender service:")
        print("   cd agent_recommender")
        print("   python main.py")
        return
    
    # Run all tests
    test_recommendations()
    test_agent_details()
    test_system_stats()
    test_model_comparison()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\n📚 For interactive API docs, visit: http://localhost:8001/docs")

if __name__ == "__main__":
    main()
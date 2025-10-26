"""
Unit tests for Agent Finder API
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models import AgentSearchRequest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_root(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestAgentSearch:
    """Test agent search functionality."""
    
    def test_search_agents_basic(self):
        """Test basic agent search."""
        request_data = {
            "user_type": "buyer",
            "state": "CA",
            "city": "Los Angeles",
            "max_results": 5
        }
        
        response = client.post("/api/v1/agents/search", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "recommendations" in data
        assert len(data["recommendations"]) <= 5
    
    def test_search_agents_with_preferences(self):
        """Test agent search with user preferences."""
        request_data = {
            "user_type": "seller",
            "state": "NY",
            "city": "New York",
            "min_price": 500000,
            "max_price": 2000000,
            "property_type": "condo",
            "is_urgent": True,
            "sub_score_preferences": {
                "responsiveness": 0.4,
                "negotiation": 0.3,
                "professionalism": 0.2,
                "market_expertise": 0.1
            },
            "max_results": 10
        }
        
        response = client.post("/api/v1/agents/search", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "recommendations" in data
    
    def test_search_agents_invalid_state(self):
        """Test agent search with invalid state."""
        request_data = {
            "user_type": "buyer",
            "state": "XX",  # Invalid state
            "city": "Test City",
            "max_results": 5
        }
        
        response = client.post("/api/v1/agents/search", json=request_data)
        # Should still return 200 but with 0 results
        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 0
    
    def test_search_agents_price_validation(self):
        """Test price range validation."""
        request_data = {
            "user_type": "buyer",
            "state": "CA",
            "city": "Los Angeles",
            "min_price": 1000000,
            "max_price": 500000,  # Invalid: max < min
            "max_results": 5
        }
        
        response = client.post("/api/v1/agents/search", json=request_data)
        assert response.status_code == 422  # Validation error


class TestStatisticalFunctions:
    """Test statistical utility functions."""
    
    def test_wilson_lower_bound(self):
        """Test Wilson Lower Bound calculation."""
        from utils.stats import wilson_lower_bound
        
        # Agent with 100% positive (but low count) should have lower bound < 1.0
        score = wilson_lower_bound(5, 5)
        assert 0 < score < 1.0
        assert score < 0.75  # Should have uncertainty
        
        # Agent with 95% positive and high count should have higher bound
        score2 = wilson_lower_bound(95, 100)
        assert score2 > 0.85
    
    def test_bayesian_rating_shrinkage(self):
        """Test Bayesian rating shrinkage."""
        from utils.stats import bayesian_rating_shrinkage
        
        # Agent with 1 review of 5 stars should be shrunk toward prior
        shrunk = bayesian_rating_shrinkage(5.0, 1, prior_mean=4.0, prior_count=10)
        assert 4.0 < shrunk < 5.0
        
        # Agent with many reviews should retain original rating
        shrunk2 = bayesian_rating_shrinkage(4.5, 100, prior_mean=4.0, prior_count=10)
        assert abs(shrunk2 - 4.5) < 0.1
    
    def test_exponential_decay(self):
        """Test exponential decay scoring."""
        from utils.stats import exponential_decay_score
        
        # Recent (0 days) should be close to 1.0
        score_recent = exponential_decay_score(0)
        assert score_recent > 0.99
        
        # Old (365 days) should be much lower
        score_old = exponential_decay_score(365)
        assert score_old < 0.1


class TestSentimentAnalysis:
    """Test sentiment analysis."""
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection."""
        from models.sentiment import sentiment_analyzer
        
        comment = "This agent was absolutely amazing! Very helpful and professional."
        sentiment, confidence = sentiment_analyzer.analyze_comment(comment)
        
        assert sentiment == "positive"
        assert confidence > 0.5
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection."""
        from models.sentiment import sentiment_analyzer
        
        comment = "Terrible experience. Agent was unresponsive and rude."
        sentiment, confidence = sentiment_analyzer.analyze_comment(comment)
        
        assert sentiment == "negative"
        assert confidence > 0.5
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        from models.sentiment import sentiment_analyzer
        
        comment = "The agent helped with the transaction."
        sentiment, confidence = sentiment_analyzer.analyze_comment(comment)
        
        assert sentiment in ["neutral", "positive"]


class TestSkillExtraction:
    """Test skill extraction."""
    
    def test_communication_skill(self):
        """Test communication skill detection."""
        from models.skills import skill_extractor
        
        comment = "Agent had excellent communication skills and was very responsive."
        skills = skill_extractor.extract_skills_from_comment(comment)
        
        assert len(skills) > 0
        assert any('communication' in skill for skill in skills)
    
    def test_negative_skill(self):
        """Test negative skill detection."""
        from models.skills import skill_extractor
        
        comment = "Agent was pushy and unresponsive to my needs."
        skills = skill_extractor.extract_skills_from_comment(comment)
        
        assert len(skills) > 0
        # Should detect negative qualities


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_diversity_metrics(self):
        """Test diversity calculation."""
        from utils.evaluation import evaluator
        
        recommendations = [
            {"advertiser_id": 1, "office_name": "Office A", "experience_years": 5},
            {"advertiser_id": 2, "office_name": "Office B", "experience_years": 10},
            {"advertiser_id": 3, "office_name": "Office A", "experience_years": 15},
        ]
        
        metrics = evaluator.calculate_diversity_metrics(recommendations)
        
        assert "diversity_office_name" in metrics
        assert metrics["diversity_office_name"] == 2/3  # 2 unique offices out of 3


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
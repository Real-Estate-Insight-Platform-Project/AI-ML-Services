"""
Test suite for Agent Recommender System.

This module contains unit tests and integration tests for the recommender system.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_preprocessing import AgentDataLoader, FeatureExtractor
from models.baseline_scorer import BaselineAgentScorer, BaselineRecommender
from models.ml_ranker import MLAgentRanker, MLRecommender
from recommender import AgentRecommenderSystem


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample agent data
        self.sample_agent_data = [
            {
                'agentId': 1,
                'name': 'John Doe',
                'starRating': 4.5,
                'numReviews': 25,
                'pastYearDeals': 15,
                'homeTransactionsLifetime': 100,
                'transactionVolumeLifetime': 50000000,
                'primaryServiceRegions': ['Downtown', 'Midtown'],
                'propertyTypes': ['Single Family Residential', 'Condo/Co-op'],
                'dealPrices': [300000, 450000, 350000, 500000],
                'businessMarket': 'Test Market',
                'partner': False,
                'isPremier': True,
                'brokerageName': 'Test Brokerage',
                'state': 'TestState'
            },
            {
                'agentId': 2,
                'name': 'Jane Smith',
                'starRating': 4.8,
                'numReviews': 50,
                'pastYearDeals': 25,
                'homeTransactionsLifetime': 200,
                'transactionVolumeLifetime': 80000000,
                'primaryServiceRegions': ['Downtown', 'Uptown'],
                'propertyTypes': ['Single Family Residential', 'Townhouse'],
                'dealPrices': [400000, 550000, 600000, 350000, 450000],
                'businessMarket': 'Test Market',
                'partner': True,
                'isPremier': False,
                'brokerageName': 'Another Brokerage',
                'state': 'TestState'
            }
        ]
        
        self.agents_df = pd.DataFrame(self.sample_agent_data)
    
    def test_feature_extractor_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor(self.agents_df)
        self.assertIsNotNone(extractor.agents_df)
        self.assertIsNotNone(extractor.market_stats)
    
    def test_geo_overlap_calculation(self):
        """Test geographical overlap calculation."""
        extractor = FeatureExtractor(self.agents_df)
        
        # Perfect overlap
        overlap = extractor.calculate_geo_overlap(['Downtown'], ['Downtown', 'Midtown'])
        self.assertEqual(overlap, 1.0)
        
        # Partial overlap
        overlap = extractor.calculate_geo_overlap(['Downtown', 'Suburb'], ['Downtown', 'Midtown'])
        self.assertEqual(overlap, 0.5)
        
        # No overlap
        overlap = extractor.calculate_geo_overlap(['Suburb'], ['Downtown', 'Midtown'])
        self.assertEqual(overlap, 0.0)
    
    def test_price_band_match(self):
        """Test price band matching."""
        extractor = FeatureExtractor(self.agents_df)
        
        # All prices within range
        deal_prices = [400000, 450000, 350000]
        match = extractor.calculate_price_band_match(400000, deal_prices, delta=0.15)
        self.assertGreater(match, 0.8)  # Should be high match
        
        # No prices within range
        deal_prices = [100000, 150000, 200000]
        match = extractor.calculate_price_band_match(500000, deal_prices, delta=0.15)
        self.assertEqual(match, 0.0)
    
    def test_property_type_match(self):
        """Test property type matching (Jaccard similarity)."""
        extractor = FeatureExtractor(self.agents_df)
        
        # Perfect match
        user_types = ['Single Family Residential']
        agent_types = ['Single Family Residential', 'Condo/Co-op']
        match = extractor.calculate_property_type_match(user_types, agent_types)
        self.assertGreater(match, 0.0)
        
        # No match
        user_types = ['Commercial']
        agent_types = ['Single Family Residential', 'Condo/Co-op']
        match = extractor.calculate_property_type_match(user_types, agent_types)
        self.assertEqual(match, 0.0)


class TestBaselineScorer(unittest.TestCase):
    """Test baseline scoring functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_features = pd.DataFrame({
            'agentId': [1, 2, 3],
            'name': ['Agent 1', 'Agent 2', 'Agent 3'],
            'geo_overlap': [1.0, 0.5, 0.0],
            'price_band_match': [0.8, 0.6, 0.2],
            'property_type_match': [1.0, 0.5, 0.3],
            'normalized_recency': [0.9, 0.7, 0.1],
            'rating_score': [0.8, 0.9, 0.6],
            'log_reviews_normalized': [0.7, 0.8, 0.3],
            'partner_premier_boost': [0.5, 1.0, 0.0]
        })
    
    def test_baseline_scorer_initialization(self):
        """Test BaselineAgentScorer initialization."""
        scorer = BaselineAgentScorer()
        self.assertIsNotNone(scorer.weights)
        
        # Check weight sum
        total_weight = sum(scorer.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_baseline_score_calculation(self):
        """Test baseline score calculation."""
        scorer = BaselineAgentScorer()
        scores = scorer.calculate_baseline_score(self.sample_features)
        
        # Check that scores are calculated
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0 <= score <= 1 for score in scores))
        
        # Agent 1 should have highest score (all features are best)
        self.assertEqual(scores.idxmax(), 0)
    
    def test_score_breakdown(self):
        """Test detailed score breakdown."""
        scorer = BaselineAgentScorer()
        breakdown = scorer.get_score_breakdown(self.sample_features, [0, 1])
        
        self.assertEqual(len(breakdown), 2)
        self.assertIn('total_score', breakdown.columns)
        self.assertIn('geo_overlap_weighted', breakdown.columns)


class TestMLRanker(unittest.TestCase):
    """Test ML ranking functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create larger sample for ML testing
        np.random.seed(42)
        n_agents = 100
        
        self.sample_data = pd.DataFrame({
            'agentId': range(n_agents),
            'name': [f'Agent {i}' for i in range(n_agents)],
            'starRating': np.random.uniform(3.0, 5.0, n_agents),
            'numReviews': np.random.poisson(20, n_agents),
            'pastYearDeals': np.random.poisson(10, n_agents),
            'homeTransactionsLifetime': np.random.poisson(50, n_agents),
            'transactionVolumeLifetime': np.random.uniform(1000000, 50000000, n_agents),
            'businessMarket': np.random.choice(['Market A', 'Market B'], n_agents),
            'partner': np.random.choice([True, False], n_agents),
            'isPremier': np.random.choice([True, False], n_agents),
            'dealPrices': [[300000, 400000, 500000] for _ in range(n_agents)],
            'primaryServiceRegions': [['Region 1', 'Region 2'] for _ in range(n_agents)],
            'propertyTypes': [['Single Family Residential'] for _ in range(n_agents)]
        })
    
    def test_ml_ranker_initialization(self):
        """Test MLAgentRanker initialization."""
        ranker = MLAgentRanker('lightgbm')
        self.assertEqual(ranker.model_type, 'lightgbm')
        self.assertFalse(ranker.is_trained)
    
    def test_feature_engineering(self):
        """Test ML feature engineering."""
        ranker = MLAgentRanker('lightgbm')
        engineered_df = ranker.feature_engineer.engineer_ml_features(self.sample_data)
        
        # Check that new features were added
        self.assertIn('rating_x_reviews', engineered_df.columns)
        self.assertIn('experience_score', engineered_df.columns)


class TestRecommenderSystem(unittest.TestCase):
    """Test complete recommender system."""
    
    def setUp(self):
        """Set up test data."""
        # Create temporary test data
        self.test_data_path = Path("test_data")
        self.test_data_path.mkdir(exist_ok=True)
        
        # Create sample state file
        sample_state_data = {
            "data": [
                {
                    "agentId": 1,
                    "name": "Test Agent 1",
                    "starRating": 4.5,
                    "numReviews": 25,
                    "pastYearDeals": 15,
                    "homeTransactionsLifetime": 100,
                    "transactionVolumeLifetime": 50000000,
                    "primaryServiceRegions": ["Downtown", "Midtown"],
                    "propertyTypes": ["Single Family Residential"],
                    "dealPrices": [300000, 450000, 350000],
                    "businessMarket": "Test Market",
                    "partner": False,
                    "isPremier": True,
                    "brokerageName": "Test Brokerage",
                    "email": "test@example.com",
                    "phoneNumber": "123-456-7890",
                    "servesOffers": True,
                    "servesListings": True,
                    "isActive": True,
                    "profileContactEnabled": True
                },
                {
                    "agentId": 2,
                    "name": "Test Agent 2",
                    "starRating": 4.8,
                    "numReviews": 50,
                    "pastYearDeals": 25,
                    "homeTransactionsLifetime": 200,
                    "transactionVolumeLifetime": 80000000,
                    "primaryServiceRegions": ["Downtown", "Uptown"],
                    "propertyTypes": ["Single Family Residential", "Condo/Co-op"],
                    "dealPrices": [400000, 550000, 600000],
                    "businessMarket": "Test Market",
                    "partner": True,
                    "isPremier": False,
                    "brokerageName": "Another Brokerage",
                    "email": "test2@example.com",
                    "phoneNumber": "123-456-7891",
                    "servesOffers": True,
                    "servesListings": True,
                    "isActive": True,
                    "profileContactEnabled": True
                }
            ]
        }
        
        with open(self.test_data_path / "TestState.json", 'w') as f:
            import json
            json.dump(sample_state_data, f)
    
    def tearDown(self):
        """Clean up test data."""
        import shutil
        if self.test_data_path.exists():
            shutil.rmtree(self.test_data_path)
    
    def test_system_initialization(self):
        """Test recommender system initialization."""
        system = AgentRecommenderSystem(str(self.test_data_path))
        system.initialize()
        
        self.assertTrue(system.is_initialized)
        self.assertIsNotNone(system.agents_df)
        self.assertGreater(len(system.agents_df), 0)
    
    def test_baseline_recommendations(self):
        """Test baseline recommendations."""
        system = AgentRecommenderSystem(str(self.test_data_path))
        system.initialize()
        
        user_query = {
            'regions': ['Downtown'],
            'budget': 400000,
            'property_types': ['Single Family Residential']
        }
        
        result = system.recommend(user_query, model_type='baseline', top_k=2)
        
        self.assertIn('recommendations', result)
        self.assertLessEqual(len(result['recommendations']), 2)
        self.assertEqual(result['model_type'], 'baseline')
    
    def test_query_validation(self):
        """Test user query validation."""
        system = AgentRecommenderSystem(str(self.test_data_path))
        
        # Test with invalid query
        invalid_query = {
            'regions': 'Downtown',  # Should be list
            'budget': 'invalid',    # Should be number
            'property_types': 'Single Family Residential'  # Should be list
        }
        
        validated = system._validate_user_query(invalid_query)
        
        self.assertIsInstance(validated['regions'], list)
        self.assertIsInstance(validated['budget'], (int, float))
        self.assertIsInstance(validated['property_types'], list)
    
    def test_agent_details(self):
        """Test getting agent details."""
        system = AgentRecommenderSystem(str(self.test_data_path))
        system.initialize()
        
        details = system.get_agent_details(1)
        
        self.assertIn('agentId', details)
        self.assertEqual(details['agentId'], 1)
        self.assertIn('name', details)
        self.assertIn('statistics', details)
    
    def test_system_stats(self):
        """Test getting system statistics."""
        system = AgentRecommenderSystem(str(self.test_data_path))
        system.initialize()
        
        stats = system.get_system_stats()
        
        self.assertIn('total_agents', stats)
        self.assertIn('unique_states', stats)
        self.assertIn('models_available', stats)
        self.assertTrue(stats['models_available']['baseline'])


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics for the recommender system."""
    
    def test_recommendation_diversity(self):
        """Test recommendation diversity metrics."""
        # Sample recommendations
        recommendations = [
            {'agentId': 1, 'businessMarket': 'Market A', 'brokerageName': 'Brokerage 1'},
            {'agentId': 2, 'businessMarket': 'Market B', 'brokerageName': 'Brokerage 2'},
            {'agentId': 3, 'businessMarket': 'Market A', 'brokerageName': 'Brokerage 1'},
        ]
        
        # Calculate market diversity
        markets = [rec['businessMarket'] for rec in recommendations]
        unique_markets = len(set(markets))
        market_diversity = unique_markets / len(markets)
        
        # Calculate brokerage diversity
        brokerages = [rec['brokerageName'] for rec in recommendations]
        unique_brokerages = len(set(brokerages))
        brokerage_diversity = unique_brokerages / len(brokerages)
        
        self.assertGreater(market_diversity, 0.0)
        self.assertLessEqual(market_diversity, 1.0)
        self.assertGreater(brokerage_diversity, 0.0)
        self.assertLessEqual(brokerage_diversity, 1.0)
    
    def test_score_distribution(self):
        """Test score distribution analysis."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        
        # Calculate score statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        score_range = np.max(scores) - np.min(scores)
        
        # Test that scores are well-distributed
        self.assertGreater(std_score, 0.1)  # Some variation
        self.assertGreater(score_range, 0.5)  # Good range


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
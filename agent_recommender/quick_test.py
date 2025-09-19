"""
Quick test of the Agent Recommender System with sample data.
"""

import sys
import os
from pathlib import Path

# Add the agent_recommender directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from utils.data_preprocessing import AgentDataLoader, FeatureExtractor
from models.baseline_scorer import BaselineRecommender


def test_with_sample_data():
    """Test the system with a small sample of real data."""
    print("🔄 Testing Agent Recommender System...")
    
    # Path to the statewise data
    data_path = current_dir.parent / "agent_data" / "statewise_data"
    
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        return False
    
    try:
        # Load just one state for quick testing
        alabama_file = data_path / "Alabama.json"
        
        if not alabama_file.exists():
            print(f"❌ Alabama.json not found in {data_path}")
            return False
        
        print(f"✅ Found data file: {alabama_file}")
        
        # Initialize data loader
        loader = AgentDataLoader(data_path)
        
        # Load and preprocess data (just Alabama for quick test)
        import json
        with open(alabama_file, 'r') as f:
            alabama_data = json.load(f)
        
        if 'data' not in alabama_data:
            print("❌ Invalid data format in Alabama.json")
            return False
        
        agents_data = alabama_data['data'][:10]  # Just first 10 agents for testing
        
        # Add state info
        for agent in agents_data:
            agent['state'] = 'Alabama'
        
        import pandas as pd
        agents_df = pd.DataFrame(agents_data)
        agents_df = loader.preprocess_agents(agents_df)
        
        print(f"✅ Loaded {len(agents_df)} agents for testing")
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor(agents_df)
        
        # Initialize baseline recommender
        baseline_recommender = BaselineRecommender(feature_extractor)
        
        # Test query
        user_query = {
            'regions': ['Fort Myers', 'Naples'],
            'budget': 500000,
            'property_types': ['Single Family Residential']
        }
        
        print(f"🔍 Testing query: {user_query}")
        
        # Get recommendations
        result = baseline_recommender.recommend_agents(user_query, top_k=5, explain=True)
        
        if 'recommendations' in result:
            print(f"✅ Successfully generated {len(result['recommendations'])} recommendations!")
            
            print("\n🏆 TOP RECOMMENDATIONS:")
            for i, agent in enumerate(result['recommendations'][:3], 1):
                print(f"{i}. {agent['name']} - Score: {agent['score']:.3f}")
                print(f"   Rating: {agent['starRating']:.1f}/5.0 ({agent['numReviews']} reviews)")
                print(f"   Deals: {agent['pastYearDeals']} in past year")
            
            # Show scoring breakdown
            if 'explanation' in result:
                print(f"\n📊 SCORING METHODOLOGY:")
                weights = result['explanation']['methodology']['weights']
                for feature, weight in weights.items():
                    print(f"   • {feature}: {weight:.3f}")
            
            return True
        else:
            print(f"❌ No recommendations generated. Result: {result}")
            return False
    
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_sample_data()
    
    if success:
        print("\n🎉 AGENT RECOMMENDER SYSTEM TEST PASSED!")
        print("✅ The system is working correctly with your data.")
    else:
        print("\n❌ AGENT RECOMMENDER SYSTEM TEST FAILED!")
        print("💡 Check the error messages above for debugging.")
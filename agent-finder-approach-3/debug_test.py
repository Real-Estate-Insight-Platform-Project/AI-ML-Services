#!/usr/bin/env python3
"""Debug test for personalization issues."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.agent_finder import AgentFinder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def test_personalization():
    """Test personalization with varied preferences."""
    
    print("🔍 Testing Agent Finder Personalization")
    print("=" * 50)
    
    # Initialize system
    af = AgentFinder('agents.csv', 'reviews.csv')
    result = af.train_system(use_cache=True)
    
    print(f"✅ Training completed: {result['success']}")
    
    # Test 1: Neutral preferences (should give alpha ≈ 0)
    neutral_prefs = {
        'responsiveness': 0.5,
        'negotiation': 0.5, 
        'professionalism': 0.5,
        'market_expertise': 0.5
    }
    
    print("\n🧪 Test 1: Neutral Preferences")
    print(f"Preferences: {neutral_prefs}")
    
    try:
        rec_result = af.recommend_agents(
            neutral_prefs, 
            {'state': 'AK', 'active_only': False}, 
            top_k=2, 
            include_explanations=False
        )
        
        alpha = rec_result.get('metadata', {}).get('fusion_alpha', 'N/A')
        recs = len(rec_result.get('recommendations', []))
        utility = rec_result['recommendations'][0]['utility_score'] if recs > 0 else 'N/A'
        
        print(f"→ Alpha: {alpha}")
        print(f"→ Recommendations: {recs}")
        print(f"→ First utility score: {utility}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Varied preferences (should give alpha > 0)
    varied_prefs = {
        'responsiveness': 0.8,  # +0.3 deviation
        'negotiation': 0.3,     # -0.2 deviation  
        'professionalism': 0.9, # +0.4 deviation
        'market_expertise': 0.6 # +0.1 deviation
    }
    
    print("\n🧪 Test 2: Varied Preferences")
    print(f"Preferences: {varied_prefs}")
    expected_alpha = sum([abs(v - 0.5) for v in varied_prefs.values()]) / len(varied_prefs) * 2
    print(f"Expected alpha: {expected_alpha:.3f}")
    
    try:
        rec_result = af.recommend_agents(
            varied_prefs,
            {'state': 'AK', 'active_only': False},
            top_k=2,
            include_explanations=False
        )
        
        alpha = rec_result.get('metadata', {}).get('fusion_alpha', 'N/A') 
        recs = len(rec_result.get('recommendations', []))
        utility = rec_result['recommendations'][0]['utility_score'] if recs > 0 else 'N/A'
        
        print(f"→ Alpha: {alpha}")
        print(f"→ Recommendations: {recs}")
        print(f"→ First utility score: {utility}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_personalization()
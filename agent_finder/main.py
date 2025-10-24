"""
Main entry point for Agent Finder system.

Provides a complete example of training and using the system.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from src.agent_finder import AgentFinder
except ImportError:
    # Fallback for different directory structures
    sys.path.append(str(Path(__file__).parent / "src"))
    from agent_finder import AgentFinder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main entry point demonstrating the Agent Finder system."""
    
    print("🏠 Agent Finder System - Complete ML-Powered Agent Recommendation")
    print("="*70)
    
    # Data files
    agents_file = "agents.csv"
    reviews_file = "reviews.csv"
    
    # Check if files exist
    if not os.path.exists(agents_file) or not os.path.exists(reviews_file):
        print(f"❌ Data files not found: {agents_file}, {reviews_file}")
        print("Please ensure the CSV files are in the current directory.")
        return
    
    print(f"📁 Found data files: {agents_file}, {reviews_file}")
    
    # Initialize system
    print("\n🚀 Initializing Agent Finder system...")
    agent_finder = AgentFinder(
        agents_file=agents_file,
        reviews_file=reviews_file,
        cache_dir="./cache"
    )
    
    # Train system
    print("\n🧠 Training the complete ML pipeline...")
    try:
        training_result = agent_finder.train_system(use_cache=True, save_cache=True)
        
        if training_result['success']:
            print("✅ Training completed successfully!")
            print(f"   - Training time: {training_result['training_time_seconds']:.1f} seconds")
            print(f"   - Agents processed: {training_result['metrics']['n_agents']}")
            print(f"   - Reviews processed: {training_result['metrics']['n_reviews']}")
            print(f"   - Themes discovered: {training_result['metrics']['n_themes']}")
            print(f"   - Model R²: {training_result['metrics']['model_performance'].get('r2', 0):.3f}")
        else:
            print("❌ Training failed!")
            return
            
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        return
    
    # Example recommendation
    print("\n🎯 Testing agent recommendations...")
    
    # Example user preferences (high professionalism, moderate others)
    user_preferences = {
        'responsiveness': 0.6,
        'negotiation': 0.5,
        'professionalism': 0.9,  # Very important
        'market_expertise': 0.7
    }
    
    # Example filters (Alaska, buying)
    user_filters = {
        'state': 'AK',
        'transaction_type': 'buying',
        'min_rating': 4.0,
        'active_only': True
    }
    
    try:
        results = agent_finder.recommend_agents(
            user_preferences=user_preferences,
            user_filters=user_filters,
            top_k=5,
            include_explanations=True
        )
        
        print("✅ Recommendations generated successfully!")
        print(f"   - Found {len(results['recommendations'])} recommendations")
        print(f"   - Filtered from {results['summary']['total_candidates']} total agents")
        print(f"   - Personalization level: {results['summary']['preference_personalization']}")
        
        # Show top recommendations
        print("\n🏆 Top Agent Recommendations:")
        print("-" * 60)
        
        for i, rec in enumerate(results['recommendations'][:3], 1):
            agent = rec
            print(f"{i}. {agent['name']} (ID: {agent['agent_id']})")
            print(f"   📍 Location: {agent['profile']['city']}, {agent['profile']['state']}")
            print(f"   ⭐ Rating: {agent['profile']['rating']:.1f} ({agent['profile']['review_count']} reviews)")
            print(f"   💼 Experience: {agent['profile']['experience_years']:.0f} years")
            print(f"   📊 Utility Score: {agent['utility_score']:.3f}")
            print(f"   🎯 Confidence: {agent['confidence_score']:.2f}")
            
            if results['explanations'] and i <= len(results['explanations']):
                explanation = results['explanations'][i-1]
                print(f"   💡 Why recommended: {explanation.get('why_recommended', 'Good overall match')}")
            
            print()
        
        # Show system statistics
        stats = agent_finder.get_system_stats()
        print("📊 System Statistics:")
        print(f"   - Total agents: {stats['data_stats']['n_agents']}")
        print(f"   - Agents with reviews: {stats['data_stats']['agents_with_reviews']}")
        print(f"   - Themes discovered: {stats['theme_stats']['n_themes']}")
        print(f"   - Model features: {stats['model_stats']['n_features']}")
        
        print("\n🎉 Agent Finder system demonstration completed!")
        print("\n💡 To start the API server, run:")
        print("   uvicorn api:app --reload --host 0.0.0.0 --port 8003")
        
    except Exception as e:
        print(f"❌ Recommendation error: {str(e)}")
        return

if __name__ == "__main__":
    main()
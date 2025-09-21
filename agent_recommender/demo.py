"""
Agent Recommender System Demo

This script demonstrates the complete Top-K Agent Recommender system 
with both baseline and ML approaches.
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent))

from utils.data_preprocessing import AgentDataLoader, FeatureExtractor
from models.baseline_scorer import BaselineRecommender, BaselineAgentScorer
from models.ml_ranker import MLRecommender, MLAgentRanker
from recommender import AgentRecommenderSystem


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_agent_summary(agent):
    """Print a summary of an agent."""
    return (
        f"🏠 {agent['name']} (ID: {agent['agentId']})\n"
        f"   ⭐ {agent['starRating']:.1f}/5.0 ({agent['numReviews']} reviews)\n"
        f"   📊 Score: {agent['score']:.3f}\n"
        f"   🏢 {agent['brokerageName']}\n"
        f"   📍 {agent['businessMarket']}\n"
        f"   📞 {agent.get('phoneNumber', 'N/A')}\n"
        f"   📧 {agent.get('email', 'N/A')}\n"
        f"   🏘️  Regions: {', '.join(agent.get('primaryServiceRegions', [])[:3])}...\n"
        f"   🏘️  Property Types: {', '.join(agent.get('propertyTypes', [])[:2])}...\n"
    )


def demo_baseline_recommender():
    """Demonstrate the baseline recommender."""
    print_section("BASELINE RECOMMENDER DEMO")
    
    # Path to your statewise data
    data_path = "agent_data/statewise_data"
    
    print("🔄 Loading agent data...")
    start_time = time.time()
    
    # Initialize data loader and feature extractor
    loader = AgentDataLoader(data_path)
    agents_df = loader.load_all_agents()
    agents_df = loader.preprocess_agents(agents_df)
    
    feature_extractor = FeatureExtractor(agents_df)
    baseline_recommender = BaselineRecommender(feature_extractor)
    
    load_time = time.time() - start_time
    print(f"✅ Loaded {len(agents_df)} agents in {load_time:.2f} seconds")
    print(f"📊 States: {agents_df['state'].nunique()}, Markets: {agents_df['businessMarket'].nunique()}")
    
    # Demo user queries
    user_queries = [
        {
            'description': "Florida luxury home buyer",
            'query': {
                'regions': ['Fort Myers', 'Naples', 'Bonita Springs-Estero'],
                'budget': 800000,
                'property_types': ['Single Family Residential', 'Condo/Co-op']
            }
        },
        {
            'description': "Texas first-time buyer",
            'query': {
                'regions': ['Dallas', 'Austin', 'Houston'],
                'budget': 350000,
                'property_types': ['Single Family Residential', 'Townhouse']
            }
        },
        {
            'description': "California investor",
            'query': {
                'regions': ['San Diego', 'Los Angeles', 'San Francisco'],
                'budget': 1200000,
                'property_types': ['Multi-Family (2-4 Unit)', 'Single Family Residential']
            }
        }
    ]
    
    for i, user_case in enumerate(user_queries, 1):
        print(f"\n📋 QUERY {i}: {user_case['description']}")
        print(f"   Regions: {user_case['query']['regions']}")
        print(f"   Budget: ${user_case['query']['budget']:,}")
        print(f"   Property Types: {user_case['query']['property_types']}")
        
        start_time = time.time()
        result = baseline_recommender.recommend_agents(
            user_case['query'], top_k=5, explain=True
        )
        query_time = time.time() - start_time
        
        print(f"\n⏱️  Query processed in {query_time:.3f} seconds")
        print(f"📊 Evaluated {result['total_agents_evaluated']} agents")
        
        if result.get('recommendations'):
            print(f"\n🏆 TOP 5 RECOMMENDED AGENTS:")
            for j, agent in enumerate(result['recommendations'], 1):
                print(f"\n{j}. {print_agent_summary(agent)}")
        
        # Show explanation for first query
        if i == 1 and result.get('explanation'):
            print("\n📖 SCORING METHODOLOGY:")
            methodology = result['explanation']['methodology']
            print(f"   Formula: {methodology['formula']}")
            print("\n   Feature Weights:")
            for feature, weight in methodology['weights'].items():
                print(f"   • {feature}: {weight:.3f}")
            
            print(f"\n📈 FEATURE STATISTICS:")
            for feature, stats in result['explanation']['feature_statistics'].items():
                print(f"   • {feature}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")


def demo_ml_recommender():
    """Demonstrate the ML recommender."""
    print_section("MACHINE LEARNING RECOMMENDER DEMO")
    
    # Path to your statewise data
    data_path = "agent_data/statewise_data"
    
    print("🔄 Loading agent data and training ML model...")
    start_time = time.time()
    
    # Initialize system
    system = AgentRecommenderSystem(data_path)
    system.initialize()
    
    init_time = time.time() - start_time
    print(f"✅ System initialized in {init_time:.2f} seconds")
    
    if system.ml_model_available:
        print("🤖 ML model is available!")
        
        # Demo query
        user_query = {
            'regions': ['Fort Myers', 'Naples'],
            'budget': 600000,
            'property_types': ['Single Family Residential']
        }
        
        print(f"\n📋 COMPARING MODELS:")
        print(f"   Regions: {user_query['regions']}")
        print(f"   Budget: ${user_query['budget']:,}")
        print(f"   Property Types: {user_query['property_types']}")
        
        # Get recommendations from both models
        comparison = system.compare_models(user_query, top_k=5)
        
        print(f"\n🔍 MODEL COMPARISON RESULTS:")
        if 'overlap_analysis' in comparison:
            overlap = comparison['overlap_analysis']
            print(f"   • Agents in common: {overlap['agents_in_common']}/5")
            print(f"   • Overlap percentage: {overlap['overlap_percentage']:.1f}%")
            print(f"   • Baseline unique: {overlap['baseline_unique']}")
            print(f"   • ML unique: {overlap['ml_unique']}")
        
        # Show top agents from each model
        print(f"\n🏆 BASELINE MODEL TOP 3:")
        baseline_recs = comparison['baseline'].get('recommendations', [])[:3]
        for i, agent in enumerate(baseline_recs, 1):
            print(f"{i}. {agent['name']} (Score: {agent['score']:.3f})")
        
        if 'ml' in comparison:
            print(f"\n🤖 ML MODEL TOP 3:")
            ml_recs = comparison['ml'].get('recommendations', [])[:3]
            for i, agent in enumerate(ml_recs, 1):
                print(f"{i}. {agent['name']} (Score: {agent['score']:.3f})")
            
            # Show feature importance if available
            ml_explanation = comparison['ml'].get('explanation', {})
            if 'feature_importance' in ml_explanation:
                print(f"\n📊 TOP ML FEATURES:")
                for feat in ml_explanation['feature_importance'][:5]:
                    print(f"   • {feat['feature']}: {feat['importance']:.3f}")
    
    else:
        print("⚠️  ML model not available. Using baseline only.")


def demo_complete_system():
    """Demonstrate the complete recommender system."""
    print_section("COMPLETE SYSTEM DEMO")
    
    # Path to your statewise data
    data_path = "agent_data/statewise_data"
    
    print("🔄 Initializing complete recommender system...")
    system = AgentRecommenderSystem(data_path)
    system.initialize()
    
    # Show system statistics
    stats = system.get_system_stats()
    print(f"\n📊 SYSTEM STATISTICS:")
    print(f"   • Total agents: {stats['total_agents']:,}")
    print(f"   • States covered: {stats['unique_states']}")
    print(f"   • Business markets: {stats['unique_markets']}")
    print(f"   • Brokerages: {stats['unique_brokerages']}")
    print(f"   • Average star rating: {stats['data_summary']['avg_star_rating']:.2f}")
    print(f"   • Average reviews per agent: {stats['data_summary']['avg_reviews']:.1f}")
    print(f"   • Total transaction volume: ${stats['data_summary']['total_transaction_volume']:,.0f}")
    
    # Demo different model types
    user_query = {
        'regions': ['Miami', 'Fort Lauderdale'],
        'budget': 500000,
        'property_types': ['Condo/Co-op']
    }
    
    print(f"\n📋 DEMO QUERY:")
    print(f"   Regions: {user_query['regions']}")
    print(f"   Budget: ${user_query['budget']:,}")
    print(f"   Property Types: {user_query['property_types']}")
    
    model_types = ['baseline']
    if system.ml_model_available:
        model_types.extend(['ml', 'ensemble'])
    
    results = {}
    for model_type in model_types:
        print(f"\n🔍 {model_type.upper()} MODEL RESULTS:")
        result = system.recommend(user_query, model_type=model_type, top_k=3, explain=False)
        results[model_type] = result
        
        if 'recommendations' in result:
            for i, agent in enumerate(result['recommendations'], 1):
                print(f"{i}. {agent['name']} - Score: {agent['score']:.3f}")
        elif 'error' in result:
            print(f"   Error: {result['error']}")
    
    # Demo agent details
    if results.get('baseline', {}).get('recommendations'):
        top_agent = results['baseline']['recommendations'][0]
        agent_id = top_agent['agentId']
        
        print(f"\n👤 DETAILED AGENT PROFILE (ID: {agent_id}):")
        details = system.get_agent_details(agent_id)
        
        if 'error' not in details:
            print(f"   Name: {details['name']}")
            print(f"   Rating: {details['starRating']:.1f}/5.0 ({details['numReviews']} reviews)")
            print(f"   Experience: {details['homeTransactionsLifetime']} lifetime transactions")
            print(f"   Volume: ${details['transactionVolumeLifetime']:,.0f}")
            print(f"   Recent activity: {details['pastYearDeals']} deals in past year")
            print(f"   Specialization: {len(details['propertyTypes'])} property types")
            print(f"   Service areas: {len(details['primaryServiceRegions'])} regions")
            
            stats = details.get('statistics', {})
            if stats:
                print(f"   Average deal value: ${stats['avg_transaction_value']:,.0f}")
                print(f"   Median deal price: ${stats['deal_price_median']:,.0f}")
                price_range = stats.get('deal_price_range', {})
                if price_range:
                    print(f"   Price range: ${price_range['min']:,.0f} - ${price_range['max']:,.0f}")


def run_performance_test():
    """Run performance tests."""
    print_section("PERFORMANCE TEST")
    
    data_path = "agent_data/statewise_data"
    system = AgentRecommenderSystem(data_path)
    
    print("🔄 Performance testing...")
    
    # Test queries
    test_queries = [
        {
            'regions': ['Fort Myers'],
            'budget': 400000,
            'property_types': ['Single Family Residential']
        },
        {
            'regions': ['Dallas', 'Austin'],
            'budget': 600000,
            'property_types': ['Single Family Residential', 'Townhouse']
        },
        {
            'regions': ['Los Angeles'],
            'budget': 1000000,
            'property_types': ['Condo/Co-op']
        }
    ]
    
    # Initialization time
    start_time = time.time()
    system.initialize()
    init_time = time.time() - start_time
    
    print(f"⏱️  Initialization time: {init_time:.2f} seconds")
    
    # Query performance
    total_query_time = 0
    for i, query in enumerate(test_queries, 1):
        start_time = time.time()
        result = system.recommend(query, model_type='baseline', top_k=10)
        query_time = time.time() - start_time
        total_query_time += query_time
        
        agents_found = len(result.get('recommendations', []))
        print(f"📊 Query {i}: {query_time:.3f}s - {agents_found} agents found")
    
    avg_query_time = total_query_time / len(test_queries)
    print(f"⚡ Average query time: {avg_query_time:.3f} seconds")
    print(f"🎯 Queries per second: {1/avg_query_time:.1f}")


def main():
    """Main demo function."""
    print_section("AGENT RECOMMENDER SYSTEM DEMO")
    print("🏠 Top-K Agent Recommender with Baseline + ML Models")
    print("📊 Interpretable scoring with advanced feature engineering")
    
    try:
        # Run demos
        demo_baseline_recommender()
        demo_ml_recommender()
        demo_complete_system()
        run_performance_test()
        
        print_section("DEMO COMPLETED SUCCESSFULLY! ✅")
        print("🎉 The Agent Recommender System is working correctly!")
        print("📋 You can now use it for real agent recommendations.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Data files not found. {e}")
        print("💡 Make sure the 'agent_data/statewise_data' directory exists with JSON files.")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("💡 Check that all dependencies are installed:")
        print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Database verification script.

This script checks if the Supabase tables have the expected structure
and contains sample data for testing the new endpoints.
"""

import os
from utils.database import db_client


def check_table_structure():
    """Check if tables exist and have expected columns."""
    print("Checking Supabase table structure...")
    
    try:
        # Check real_estate_agents table
        print("\n1. Checking real_estate_agents table...")
        agents_response = db_client.client.table('real_estate_agents').select('*').limit(1).execute()
        if agents_response.data:
            agent = agents_response.data[0]
            required_columns = ['advertiser_id', 'full_name', 'review_count', 'positive_review_count', 'negative_review_count', 'neutral_review_count']
            missing_columns = [col for col in required_columns if col not in agent]
            
            if missing_columns:
                print(f"❌ Missing columns in real_estate_agents: {missing_columns}")
            else:
                print(f"✅ real_estate_agents table structure looks good")
                print(f"   Sample agent: {agent.get('full_name')} (ID: {agent.get('advertiser_id')})")
        else:
            print("❌ real_estate_agents table is empty or doesn't exist")
            
        # Check reviews table  
        print("\n2. Checking reviews table...")
        reviews_response = db_client.client.table('reviews').select('*').limit(1).execute()
        if reviews_response.data:
            review = reviews_response.data[0]
            required_columns = ['review_id', 'advertiser_id', 'review_rating', 'review_comment', 'review_created_date']
            missing_columns = [col for col in required_columns if col not in review]
            
            if missing_columns:
                print(f"❌ Missing columns in reviews: {missing_columns}")
            else:
                print(f"✅ reviews table structure looks good")
                print(f"   Sample review for agent ID: {review.get('advertiser_id')}")
        else:
            print("❌ reviews table is empty or doesn't exist")
            
        # Check uszips table
        print("\n3. Checking uszips table...")
        uszips_response = db_client.client.table('uszips').select('*').limit(1).execute()
        if uszips_response.data:
            zipcode = uszips_response.data[0]
            required_columns = ['zip', 'city', 'state_name']
            missing_columns = [col for col in required_columns if col not in zipcode]
            
            if missing_columns:
                print(f"❌ Missing columns in uszips: {missing_columns}")
            else:
                print(f"✅ uszips table structure looks good") 
                print(f"   Sample: {zipcode.get('city')}, {zipcode.get('state_name')} ({zipcode.get('zip')})")
        else:
            print("❌ uszips table is empty or doesn't exist")
            
    except Exception as e:
        print(f"❌ Error checking tables: {e}")


def check_sample_data():
    """Check if we have sample data for testing."""
    print("\n" + "="*50)
    print("Checking sample data for testing...")
    
    try:
        # Find an agent with reviews for testing
        print("\n1. Looking for agents with reviews...")
        agents_with_reviews = db_client.client.table('real_estate_agents').select('advertiser_id, full_name, review_count').gt('review_count', 0).limit(5).execute()
        
        if agents_with_reviews.data:
            print("✅ Found agents with reviews for testing:")
            for agent in agents_with_reviews.data:
                print(f"   - {agent['full_name']} (ID: {agent['advertiser_id']}) - {agent['review_count']} reviews")
        else:
            print("❌ No agents found with reviews")
            
        # Check state coverage
        print("\n2. Checking state coverage...")
        states_response = db_client.client.table('uszips').select('state_name').limit(10).execute()
        if states_response.data:
            states = list(set([row['state_name'] for row in states_response.data]))
            print(f"✅ Found {len(states)} sample states: {states[:5]}...")
        else:
            print("❌ No states found in uszips table")
            
    except Exception as e:
        print(f"❌ Error checking sample data: {e}")


def main():
    """Run all checks."""
    print("Supabase Database Verification")
    print("=" * 50)
    
    # Check environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Missing environment variables:")
        print("   Please set SUPABASE_URL and SUPABASE_SERVICE_KEY")
        print("   You can copy .env.example to .env and fill in your values")
        return
    
    print(f"✅ Environment variables found")
    print(f"   Supabase URL: {supabase_url[:50]}...")
    
    check_table_structure()
    check_sample_data()
    
    print("\n" + "=" * 50)
    print("Verification complete!")


if __name__ == "__main__":
    main()
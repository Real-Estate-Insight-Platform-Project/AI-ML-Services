"""
Main data processing script to preprocess and update database with engineered features.

Run this script to:
1. Load data from Supabase
2. Analyze sentiments
3. Extract skills from reviews
4. Calculate aggregated metrics
5. Update database with processed features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utils.database import db_client
from utils.preprocessing import preprocessor
from models.sentiment import sentiment_analyzer
from models.skills import skill_extractor
from utils.geo import GeoUtils
import argparse
from pandas.api.types import is_scalar

def _is_valid_scalar(v):
    """Check if value is a true scalar (not list/Series/array/dict) and not NaN."""
    if not is_scalar(v):
        return False
    try:
        return not pd.isna(v)
    except Exception:
        return v is not None

def process_all_data(dry_run: bool = False):
    """
    Process all data and update database.
    
    Args:
        dry_run: If True, don't update database, just show results
    """
    print("=" * 80)
    print("AGENT FINDER DATA PROCESSING PIPELINE")
    print("=" * 80)
    
    # Load data
    print("\n[1/7] Loading data from Supabase...")
    agents_df = db_client.get_all_agents()
    reviews_df = db_client.get_all_reviews()
    uszips_df = db_client.get_zipcodes()
    
    print(f"  - Loaded {len(agents_df)} agents")
    print(f"  - Loaded {len(reviews_df)} reviews")
    print(f"  - Loaded {len(uszips_df)} zipcodes")
    
    # Preprocess reviews
    print("\n[2/7] Preprocessing reviews (sentiment analysis)...")

    def add_sentiment_to_reviews(reviews_df):
        """Add sentiment to reviews, using rating fallback when needed."""
        sentiments = []
        confidences = []
        
        for _, row in reviews_df.iterrows():
            comment = row.get('review_comment')
            rating = row.get('review_rating')
            
            # Use rating as fallback
            sentiment, confidence = sentiment_analyzer.analyze_comment(
                comment, 
                rating=rating
            )
            
            sentiments.append(sentiment)
            confidences.append(confidence)
        
        reviews_df['sentiment'] = sentiments
        reviews_df['sentiment_confidence'] = confidences
        
        return reviews_df

    reviews_processed = preprocessor.preprocess_reviews(reviews_df)
    reviews_processed = add_sentiment_to_reviews(reviews_processed)
    
    sentiment_counts = reviews_processed['sentiment'].value_counts()
    print(f"  - Sentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"    {sentiment}: {count} ({count/len(reviews_processed)*100:.1f}%)")
    
    # Extract skills
    print("\n[3/7] Extracting skills from review comments...")
    skill_scores = skill_extractor.aggregate_agent_skills(
        reviews_processed,
        comment_col='review_comment',
        date_col='review_created_date'
    )
    
    if not skill_scores.empty:
        skill_cols = [col for col in skill_scores.columns if col.startswith('skill_') or col.startswith('negative_')]
        print(f"  - Extracted {len(skill_cols)} skill categories")
        print(f"  - Skills for {len(skill_scores)} agents")
    
    # Aggregate review metrics
    print("\n[4/7] Aggregating review metrics...")
    review_metrics = preprocessor.aggregate_review_metrics(reviews_processed)
    print(f"  - Aggregated metrics for {len(review_metrics)} agents")
    
    # Debug: Check review metrics
    if len(review_metrics) > 0:
        total_positive = review_metrics['positive_review_count'].sum()
        total_negative = review_metrics['negative_review_count'].sum()
        print(f"  - Total positive reviews across all agents: {total_positive}")
        print(f"  - Total negative reviews across all agents: {total_negative}")
        print(f"  - Agents with reviews: {len(review_metrics[review_metrics['positive_review_count'] > 0])}")
        
        # Show top 3 agents by positive reviews
        top_agents = review_metrics.nlargest(3, 'positive_review_count')
        print(f"  - Top 3 agents by positive reviews:")
        for _, agent in top_agents.iterrows():
            print(f"    Agent {agent['advertiser_id']}: {agent['positive_review_count']} positive, {agent['negative_review_count']} negative")
    
    # Preprocess agents
    print("\n[5/7] Engineering agent features...")
    agents_processed = preprocessor.preprocess_agents(
        agents_df,
        review_metrics,
        skill_scores
    )
    
    print(f"  - Processed {len(agents_processed)} agents with all features")
    
    # Fix column names after merge (remove _x/_y suffixes)
    print("  - Fixing column names after merge...")
    columns_to_fix = [
        'positive_review_count', 'negative_review_count', 'neutral_review_count',
        'wilson_score', 'avg_sub_responsiveness', 'avg_sub_negotiation', 
        'avg_sub_professionalism', 'avg_sub_market_expertise',
        'buyer_review_count', 'seller_review_count', 'buyer_positive_count', 
        'seller_positive_count', 'buyer_satisfaction', 'seller_satisfaction',
        'skill_communication', 'skill_local_knowledge', 'skill_attention_to_detail',
        'skill_patience', 'skill_honesty', 'skill_dedication',
        'negative_unresponsive', 'negative_pushy', 'negative_unprofessional',
        'negative_inexperienced'
    ]
    
    for col in columns_to_fix:
        y_col = f"{col}_y"
        x_col = f"{col}_x"
        
        if y_col in agents_processed.columns:
            # Use the _y column (from review metrics) as the main column
            agents_processed[col] = agents_processed[y_col]
            agents_processed.drop(columns=[y_col], inplace=True)
            print(f"    - Renamed {y_col} to {col}")
        
        if x_col in agents_processed.columns:
            # Drop the _x column (from original agents)
            agents_processed.drop(columns=[x_col], inplace=True)
            print(f"    - Dropped {x_col}")
    
    print(f"  - After column cleanup: {len(agents_processed.columns)} columns")
    print(f"  - Has positive_review_count: {'positive_review_count' in agents_processed.columns}")
    
    # Show sample results
    print("\n[6/7] Sample processed agent:")
    sample_agent = agents_processed.iloc[0]
    print(f"  - Name: {sample_agent['full_name']}")
    print(f"  - Positive reviews: {sample_agent.get('positive_review_count', 0)}")
    print(f"  - Negative reviews: {sample_agent.get('negative_review_count', 0)}")
    print(f"  - Wilson score: {sample_agent.get('wilson_score', 0):.3f}")
    print(f"  - Shrunk rating: {sample_agent.get('shrunk_rating', 0):.2f}")
    print(f"  - Performance score: {sample_agent.get('performance_score', 0):.3f}")
    print(f"  - Buyer satisfaction: {sample_agent.get('buyer_satisfaction', 0):.2f}")
    print(f"  - Seller satisfaction: {sample_agent.get('seller_satisfaction', 0):.2f}")
    
    # Debug: Check if this agent actually has reviews
    agent_id = sample_agent['advertiser_id']
    agent_reviews = reviews_processed[reviews_processed['advertiser_id'] == agent_id]
    print(f"  - Agent ID: {agent_id}")
    print(f"  - Number of reviews for this agent: {len(agent_reviews)}")
    if len(agent_reviews) > 0:
        sentiment_dist = agent_reviews['sentiment'].value_counts()
        print(f"  - Sentiment breakdown: {dict(sentiment_dist)}")
    
    # Debug: Check what columns are in agents_processed
    print(f"  - Agents processed shape: {agents_processed.shape}")
    print(f"  - Available columns: {list(agents_processed.columns)}")
    print(f"  - Has positive_review_count: {'positive_review_count' in agents_processed.columns}")
    
    # Check if any agent has positive reviews (only if column exists)
    if 'positive_review_count' in agents_processed.columns:
        agents_with_reviews = agents_processed[agents_processed['positive_review_count'] > 0]
        print(f"  - Agents with positive reviews: {len(agents_with_reviews)}")
        if len(agents_with_reviews) > 0:
            print(f"  - Max positive reviews for any agent: {agents_processed['positive_review_count'].max()}")
            top_agent = agents_processed.loc[agents_processed['positive_review_count'].idxmax()]
            print(f"  - Top agent: {top_agent['full_name']} with {top_agent['positive_review_count']} positive reviews")
    else:
        print("  - ERROR: positive_review_count column missing from agents_processed!")
        print("  - This indicates a merge issue in the preprocessing pipeline")
    
    # Update database
    if not dry_run:
        print("\n[7/7] Updating database...")
        
        # Update reviews
        print("  - Updating reviews with sentiment...")
        review_updates = []
        for _, row in reviews_processed.iterrows():
            # Handle NaN values safely
            days_since_review = row['days_since_review']
            if pd.isna(days_since_review):
                days_since_review = None
            else:
                days_since_review = int(days_since_review)
            
            recency_weight = row['recency_weight']
            if pd.isna(recency_weight):
                recency_weight = None
            else:
                recency_weight = float(recency_weight)
            
            review_updates.append({
                'review_id': row['review_id'],
                'sentiment': row['sentiment'],
                'sentiment_confidence': row['sentiment_confidence'],
                'days_since_review': days_since_review,
                'recency_weight': recency_weight
            })
        
        # Batch update in chunks using safe update method
        chunk_size = 50  # Smaller chunks to avoid timeouts
        total_chunks = (len(review_updates) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(review_updates), chunk_size):
            chunk_num = (i // chunk_size) + 1
            chunk = review_updates[i:i+chunk_size]
            print(f"    Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} reviews)...")
            
            try:
                db_client.batch_update_reviews_safe(chunk)
                print(f"    ✓ Chunk {chunk_num} completed successfully")
            except Exception as e:
                print(f"    ✗ Error in chunk {chunk_num}: {e}")
                print(f"    Continuing with next chunk...")
        
        print(f"  - Updated {len(review_updates)} reviews")
        
        # Update agents
        print("  - Updating agents with processed features...")
        agent_updates = []
        
        # Columns to update
        update_columns = [
            'positive_review_count', 'negative_review_count', 'neutral_review_count',
            'wilson_score', 'shrunk_rating', 'performance_score',
            'buyer_satisfaction', 'seller_satisfaction',
            'buyer_review_count', 'seller_review_count',
            'avg_sub_responsiveness', 'avg_sub_negotiation',
            'avg_sub_professionalism', 'avg_sub_market_expertise',
            'experience_score', 'min_price', 'max_price'
        ]
        
        # Add skill columns
        skill_cols = [col for col in agents_processed.columns if col.startswith('skill_') or col.startswith('negative_')]
        update_columns.extend(skill_cols)
        
        for _, row in agents_processed.iterrows():
            update_dict = {'advertiser_id': row['advertiser_id']}
            for col in update_columns:
                if col in row.index:
                    value = row[col]
                    if _is_valid_scalar(value):
                        if isinstance(value, (int, np.integer)):
                            update_dict[col] = int(value)
                        elif isinstance(value, (float, np.floating)):
                            update_dict[col] = float(value)
                        else:
                            update_dict[col] = value

            
            # Add parsed lists as arrays (matching database schema)
            if 'property_types' in row.index and row['property_types'] is not None:
                try:
                    if hasattr(row['property_types'], '__len__') and len(row['property_types']) > 0:
                        # Store as array for PostgreSQL TEXT[] column
                        update_dict['property_types'] = row['property_types'].tolist() if hasattr(row['property_types'], 'tolist') else list(row['property_types'])
                except:
                    pass
            
            if 'additional_specializations' in row.index and row['additional_specializations'] is not None:
                try:
                    if hasattr(row['additional_specializations'], '__len__') and len(row['additional_specializations']) > 0:
                        # Store as array for PostgreSQL TEXT[] column
                        update_dict['additional_specializations'] = row['additional_specializations'].tolist() if hasattr(row['additional_specializations'], 'tolist') else list(row['additional_specializations'])
                except:
                    pass
            
            agent_updates.append(update_dict)
        
        # Batch update using safe update method
        for i in range(0, len(agent_updates), chunk_size):
            chunk = agent_updates[i:i+chunk_size]
            db_client.batch_update_agents_safe(chunk)
            if (i + chunk_size) % 500 == 0:
                print(f"    Updated {min(i+chunk_size, len(agent_updates))}/{len(agent_updates)} agents")
        
        print(f"  - Updated {len(agent_updates)} agents")
        
        print("\n" + "=" * 80)
        print("DATA PROCESSING COMPLETE!")
        print("=" * 80)
    else:
        print("\n[7/7] DRY RUN - Skipping database update")
        print("\n" + "=" * 80)
        print("DRY RUN COMPLETE - No changes made to database")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process agent finder data')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without updating database'
    )
    
    args = parser.parse_args()
    
    try:
        process_all_data(dry_run=args.dry_run)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
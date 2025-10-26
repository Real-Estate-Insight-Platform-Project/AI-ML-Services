"""
Main data processing script to preprocess and update database with engineered features.

Run this script to:
1. Load data from Supabase
2. Analyze sentiments
3. Extract skills from reviews
4. Calculate aggregated metrics
5. Update database with processed features

FIXED: Added comprehensive validation to ensure all values meet database constraints
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utils.database import db_client
from utils.preprocessing import preprocessor
from utils.validation import validate_row_for_db, validate_dataframe_column
from models.sentiment import sentiment_analyzer
from models.skills import skill_extractor
from utils.geo import GeoUtils
import argparse
from pandas.api.types import is_scalar


def process_all_data(dry_run: bool = False):
    """
    Process all data and update database.
    
    Args:
        dry_run: If True, don't update database, just show results
    """
    print("=" * 80)
    print("AGENT FINDER DATA PROCESSING PIPELINE (WITH VALIDATION)")
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
    
    # Validate review columns before proceeding
    print("  - Validating review columns...")
    for col in ['sentiment_confidence', 'recency_weight']:
        if col in reviews_processed.columns:
            reviews_processed[col] = validate_dataframe_column(reviews_processed, col)
            print(f"    ✓ {col}: validated")
    
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
        
        # Validate skill scores
        print("  - Validating skill scores...")
        for col in skill_cols:
            if col in skill_scores.columns:
                skill_scores[col] = validate_dataframe_column(skill_scores, col)
        print(f"    ✓ All skill scores validated")
    
    # Aggregate review metrics
    print("\n[4/7] Aggregating review metrics...")
    review_metrics = preprocessor.aggregate_review_metrics(reviews_processed)
    print(f"  - Aggregated metrics for {len(review_metrics)} agents")
    
    # Validate review metrics
    print("  - Validating aggregated metrics...")
    metric_cols_to_validate = ['wilson_score', 'buyer_satisfaction', 'seller_satisfaction',
                               'avg_sub_responsiveness', 'avg_sub_negotiation',
                               'avg_sub_professionalism', 'avg_sub_market_expertise']
    for col in metric_cols_to_validate:
        if col in review_metrics.columns:
            review_metrics[col] = validate_dataframe_column(review_metrics, col)
    print(f"    ✓ All review metrics validated")
    
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
    
    # CRITICAL: Validate all score columns before database insertion
    print("  - Validating agent scores (CRITICAL - prevents constraint violations)...")
    score_cols_to_validate = [
        'wilson_score', 'performance_score', 'experience_score',
        'buyer_satisfaction', 'seller_satisfaction', 'shrunk_rating',
        'avg_sub_responsiveness', 'avg_sub_negotiation',
        'avg_sub_professionalism', 'avg_sub_market_expertise'
    ]
    
    # Add all skill columns
    skill_cols = [col for col in agents_processed.columns if col.startswith('skill_') or col.startswith('negative_')]
    score_cols_to_validate.extend(skill_cols)
    
    validation_results = []
    for col in score_cols_to_validate:
        if col in agents_processed.columns:
            before_validation = agents_processed[col].describe()
            agents_processed[col] = validate_dataframe_column(agents_processed, col)
            after_validation = agents_processed[col].describe()
            
            # Check if any values were modified
            if before_validation['max'] != after_validation['max'] or before_validation['min'] != after_validation['min']:
                validation_results.append(f"    - {col}: Fixed out-of-range values (min={after_validation['min']:.4f}, max={after_validation['max']:.4f})")
            else:
                validation_results.append(f"    - {col}: All values valid")
    
    # Print validation summary
    if validation_results:
        for result in validation_results[:5]:  # Show first 5
            print(result)
        if len(validation_results) > 5:
            print(f"    - ... and {len(validation_results) - 5} more columns validated")
    
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
        
        if x_col in agents_processed.columns:
            # Drop the _x column (from original agents)
            agents_processed.drop(columns=[x_col], inplace=True)
    
    print(f"    ✓ Column cleanup complete")
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
    
    # Update database
    if not dry_run:
        print("\n[7/7] Updating database...")
        
        # Update reviews
        print("  - Updating reviews with sentiment and recency weights...")
        review_updates = []
        
        for _, row in reviews_processed.iterrows():
            # Create update dict and validate it
            update_dict = {
                'review_id': row['review_id'],
                'sentiment': row['sentiment'],
                'sentiment_confidence': float(row['sentiment_confidence']) if pd.notna(row['sentiment_confidence']) else None,
                'days_since_review': int(row['days_since_review']) if pd.notna(row['days_since_review']) else None,
                'recency_weight': float(row['recency_weight']) if pd.notna(row['recency_weight']) else None
            }
            
            # Validate the entire row
            validated_dict = validate_row_for_db(update_dict)
            review_updates.append(validated_dict)
        
        # Batch update in chunks
        chunk_size = 50
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
            
            # Add all update columns
            for col in update_columns:
                if col in row.index and pd.notna(row[col]):
                    update_dict[col] = row[col]
            
            # Add parsed lists as arrays
            if 'property_types' in row.index and row['property_types'] is not None:
                try:
                    if hasattr(row['property_types'], '__len__') and len(row['property_types']) > 0:
                        update_dict['property_types'] = row['property_types'].tolist() if hasattr(row['property_types'], 'tolist') else list(row['property_types'])
                except:
                    pass
            
            if 'additional_specializations' in row.index and row['additional_specializations'] is not None:
                try:
                    if hasattr(row['additional_specializations'], '__len__') and len(row['additional_specializations']) > 0:
                        update_dict['additional_specializations'] = row['additional_specializations'].tolist() if hasattr(row['additional_specializations'], 'tolist') else list(row['additional_specializations'])
                except:
                    pass
            
            # CRITICAL: Validate the entire row before adding to batch
            validated_dict = validate_row_for_db(update_dict)
            agent_updates.append(validated_dict)
        
        # Batch update using safe update method
        chunk_size = 50
        total_chunks = (len(agent_updates) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(agent_updates), chunk_size):
            chunk_num = (i // chunk_size) + 1
            chunk = agent_updates[i:i+chunk_size]
            print(f"    Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} agents)...")
            
            # Debug: Check performance_score values in this chunk
            perf_scores = [a.get('performance_score') for a in chunk if 'performance_score' in a]
            if perf_scores:
                print(f"      Performance scores: min={min(perf_scores):.4f}, max={max(perf_scores):.4f}")
            
            try:
                db_client.batch_update_agents_safe(chunk)
                print(f"    ✓ Chunk {chunk_num} completed successfully")
            except Exception as e:
                print(f"    ✗ Error in chunk {chunk_num}: {e}")
                # Print first agent in chunk for debugging
                if len(chunk) > 0:
                    print(f"    First agent in failed chunk: {chunk[0].get('advertiser_id')}")
                    problem_keys = [k for k in chunk[0].keys() if k in ['performance_score', 'wilson_score', 'buyer_satisfaction', 'seller_satisfaction']]
                    print(f"    Problematic values: {[(k, chunk[0][k]) for k in problem_keys]}")
                print(f"    Continuing with next chunk...")
        
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
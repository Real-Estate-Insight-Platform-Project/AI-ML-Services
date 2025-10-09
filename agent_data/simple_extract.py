#!/usr/bin/env python3
"""
Simple script to combine all statewise JSON data into a single CSV file.
"""

import os
import json
import pandas as pd
import glob

def combine_statewise_data():
    """Combine all statewise JSON data into a single CSV file."""
    
    # Directory containing the JSON files
    data_dir = "statewise_data"
    output_file = "all_agents_combined.csv"
    
    all_agents = []
    
    # Get all JSON files in the statewise_data directory
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        # Extract state name from filename
        state_name = os.path.basename(json_file).replace('.json', '').replace('_', ' ')
        
        print(f"Processing: {state_name}")
        
        try:
            # Read JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract agents data
            if 'data' in data and isinstance(data['data'], list):
                for agent in data['data']:
                    # Add state column to each agent
                    agent['state'] = state_name
                    all_agents.append(agent)
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"\nTotal agents collected: {len(all_agents)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_agents)
    
    # Move state column to the front
    if 'state' in df.columns:
        cols = ['state'] + [col for col in df.columns if col != 'state']
        df = df[cols]
    
    # Convert list columns to string representation for CSV compatibility
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n✅ Successfully saved {len(df)} agents to '{output_file}'")
    print(f"📊 States included: {df['state'].nunique()}")
    print(f"📝 Columns: {len(df.columns)}")
    
    # Show sample of data
    print(f"\nSample data (first 3 rows):")
    print(df[['state', 'name', 'email', 'starRating', 'numReviews']].head(3).to_string(index=False))
    
    return output_file

if __name__ == "__main__":
    combine_statewise_data()
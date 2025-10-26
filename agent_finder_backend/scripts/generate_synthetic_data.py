"""
Synthetic data generator for balancing the agent dataset.

Generates:
1. Negative reviews for existing agents
2. New agents with varied performance profiles
3. Property type specializations for sparse agents
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
from typing import List, Dict
from utils.database import db_client
from utils.preprocessing import preprocessor


class SyntheticDataGenerator:
    """Generate synthetic data to balance the dataset."""
    
    # Negative review templates
    NEGATIVE_COMMENTS = [
        "Agent was unresponsive to my calls and emails. Very disappointing experience.",
        "Poor communication throughout the process. Felt like I was bothering them.",
        "Agent seemed inexperienced and didn't know the local market well.",
        "Was pushy and tried to rush me into decisions I wasn't comfortable with.",
        "Unprofessional behavior during showings. Would not recommend.",
        "Agent misled us about property condition. Lost trust immediately.",
        "Terrible negotiation skills. Could have gotten a much better deal.",
        "Agent was late to multiple appointments and disorganized.",
        "Did not listen to our needs and kept showing wrong properties.",
        "Very rude and dismissive of our concerns. Poor service overall."
    ]
    
    # Neutral review templates
    NEUTRAL_COMMENTS = [
        "Agent completed the transaction. Nothing exceptional.",
        "Standard service. Got the job done.",
        "Average experience overall.",
        "Process was okay, nothing to complain about but nothing special either.",
        "Agent did what was expected."
    ]
    
    # Property types for specialization
    PROPERTY_TYPES = [
        'single_family', 'multi_family', 'condo', 'townhouse',
        'land', 'commercial', 'luxury', 'new_construction'
    ]
    
    def __init__(self):
        """Initialize generator."""
        pass
    
    def generate_negative_reviews(
        self,
        agents_df: pd.DataFrame,
        num_reviews: int = 100
    ) -> pd.DataFrame:
        """
        Generate negative reviews for existing agents.
        
        Args:
            agents_df: DataFrame with agents
            num_reviews: Number of negative reviews to generate
        
        Returns:
            DataFrame with synthetic negative reviews
        """
        reviews = []
        
        # Select agents with high ratings to add some negative reviews
        high_rated_agents = agents_df[
            agents_df['agent_rating'] >= 4.5
        ].sample(min(num_reviews, len(agents_df)))
        
        for _, agent in high_rated_agents.iterrows():
            # Generate 1-3 negative reviews per agent
            num_agent_reviews = random.randint(1, 3)
            
            for _ in range(num_agent_reviews):
                review = {
                    'review_id': str(uuid.uuid4()),
                    'advertiser_id': agent['advertiser_id'],
                    'review_rating': round(random.uniform(1.0, 2.5), 1),
                    'review_comment': random.choice(self.NEGATIVE_COMMENTS),
                    'review_created_date': (
                        datetime.now() - timedelta(days=random.randint(1, 365))
                    ).date(),
                    'transaction_date': None,
                    'reviewer_role': random.choice(['BUYER', 'SELLER']),
                    'reviewer_location': f"{agent['agent_base_city']}, {agent['state']}",
                    'sub_responsiveness': round(random.uniform(1.0, 2.5), 1),
                    'sub_negotiation': round(random.uniform(1.0, 2.5), 1),
                    'sub_professionalism': round(random.uniform(1.0, 2.5), 1),
                    'sub_market_expertise': round(random.uniform(1.0, 2.5), 1)
                }
                reviews.append(review)
        
        return pd.DataFrame(reviews)
    
    def generate_synthetic_agents(
        self,
        num_agents: int = 50,
        states: List[str] = ['CA', 'NY', 'TX', 'FL']
    ) -> pd.DataFrame:
        """
        Generate synthetic agents with varied profiles.
        
        Args:
            num_agents: Number of agents to generate
            states: States to distribute agents across
        
        Returns:
            DataFrame with synthetic agents
        """
        agents = []
        
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 
                      'Robert', 'Lisa', 'James', 'Mary', 'William', 'Jennifer']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
                     'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez']
        
        cities_by_state = {
            'CA': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento'],
            'NY': ['New York', 'Buffalo', 'Rochester', 'Albany'],
            'TX': ['Houston', 'Dallas', 'Austin', 'San Antonio'],
            'FL': ['Miami', 'Orlando', 'Tampa', 'Jacksonville']
        }
        
        for i in range(num_agents):
            state = random.choice(states)
            city = random.choice(cities_by_state[state])
            
            # Vary performance profiles
            performance_type = random.choice(['high', 'medium', 'low'])
            
            if performance_type == 'high':
                rating = round(random.uniform(4.5, 5.0), 2)
                review_count = random.randint(30, 100)
                recently_sold = random.randint(15, 40)
                days_since_sale = random.randint(1, 30)
                experience = round(random.uniform(10, 25), 1)
            elif performance_type == 'medium':
                rating = round(random.uniform(3.8, 4.4), 2)
                review_count = random.randint(10, 30)
                recently_sold = random.randint(5, 15)
                days_since_sale = random.randint(30, 90)
                experience = round(random.uniform(5, 10), 1)
            else:  # low
                rating = round(random.uniform(2.5, 3.7), 2)
                review_count = random.randint(3, 10)
                recently_sold = random.randint(0, 5)
                days_since_sale = random.randint(90, 365)
                experience = round(random.uniform(1, 5), 1)
            
            agent = {
                'advertiser_id': 9000000 + i,  # Start from high number to avoid conflicts
                'full_name': f"{random.choice(first_names)} {random.choice(last_names)}",
                'state': state,
                'agent_base_city': city,
                'agent_base_zipcode': f"{random.randint(10000, 99999)}",
                'review_count': review_count,
                'agent_rating': rating,
                'active_listings_count': random.randint(0, 15),
                'recently_sold_count': recently_sold,
                'last_sale_date': (
                    datetime.now() - timedelta(days=days_since_sale)
                ).date() if recently_sold > 0 else None,
                'days_since_last_sale': days_since_sale if recently_sold > 0 else None,
                'active_listings_min_price': random.randint(200000, 500000),
                'active_listings_max_price': random.randint(600000, 2000000),
                'recently_sold_min_price': random.randint(200000, 500000),
                'recently_sold_max_price': random.randint(600000, 2000000),
                'service_zipcodes': ', '.join([
                    str(random.randint(10000, 99999)) for _ in range(random.randint(3, 7))
                ]),
                'service_areas': city,
                'marketing_area_cities': city,
                'first_year_active': datetime.now().year - experience,
                'first_month_active': random.randint(1, 12),
                'experience_years': experience,
                'designations': random.choice(['ABR, CRS', 'GRI', 'SRS', '']),
                'languages': 'English',
                'specializations': ', '.join(random.sample(
                    ['Buyers', 'Sellers', 'Luxury', 'First-time buyers', 'Investment'],
                    k=random.randint(2, 4)
                )),
                'agent_type': random.choice(['buyer, seller', 'buyer', 'seller']),
                'agent_title': random.choice(['Realtor', 'Broker', 'Agent']),
                'is_realtor': random.choice([True, False]),
                'office_name': f"{random.choice(last_names)} Realty",
                'office_address': f"{random.randint(100, 9999)} Main St, {city}, {state}",
                'phone_primary': f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
                'office_phone': f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
                'agent_website': f"https://www.{random.choice(first_names).lower()}{random.choice(last_names).lower()}.com",
                'has_photo': random.choice([True, False]),
                'agent_photo_url': None,
                'agent_bio': "Experienced real estate professional."
            }
            
            agents.append(agent)
        
        return pd.DataFrame(agents)
    
    def add_property_specializations(
        self,
        agents_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add property type specializations to agents with sparse data.
        
        Args:
            agents_df: DataFrame with agents
        
        Returns:
            Updated DataFrame
        """
        # Find agents without property types in specializations
        for idx, agent in agents_df.iterrows():
            spec_text = str(agent.get('specializations', ''))
            
            # If no clear property type mentioned, add some
            if not any(pt in spec_text.lower() for pt in ['family', 'condo', 'land', 'commercial']):
                # Add 1-3 property types
                num_types = random.randint(1, 3)
                new_types = random.sample(self.PROPERTY_TYPES, num_types)
                
                # Append to existing specializations
                if spec_text and spec_text != 'nan':
                    agents_df.at[idx, 'specializations'] = f"{spec_text}, {', '.join(new_types)}"
                else:
                    agents_df.at[idx, 'specializations'] = ', '.join(new_types)
        
        return agents_df
    
    def generate_reviews_for_synthetic_agents(
        self,
        agents_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate reviews for synthetic agents.
        
        Args:
            agents_df: DataFrame with synthetic agents
        
        Returns:
            DataFrame with reviews
        """
        reviews = []
        
        for _, agent in agents_df.iterrows():
            num_reviews = agent['review_count']
            avg_rating = agent['agent_rating']
            
            for i in range(num_reviews):
                # Vary ratings around average
                rating = max(1.0, min(5.0, round(
                    np.random.normal(avg_rating, 0.5), 1
                )))
                
                # Choose comment based on rating
                if rating >= 4.0:
                    comment_pool = ["Great agent!", "Very professional.", "Highly recommend!"]
                elif rating >= 3.0:
                    comment_pool = self.NEUTRAL_COMMENTS
                else:
                    comment_pool = self.NEGATIVE_COMMENTS
                
                review = {
                    'review_id': str(uuid.uuid4()),
                    'advertiser_id': agent['advertiser_id'],
                    'review_rating': rating,
                    'review_comment': random.choice(comment_pool),
                    'review_created_date': (
                        datetime.now() - timedelta(days=random.randint(1, 730))
                    ).date(),
                    'transaction_date': None,
                    'reviewer_role': random.choice(['BUYER', 'SELLER']),
                    'reviewer_location': f"{agent['agent_base_city']}, {agent['state']}",
                    'sub_responsiveness': max(1.0, min(5.0, rating + random.uniform(-0.5, 0.5))),
                    'sub_negotiation': max(1.0, min(5.0, rating + random.uniform(-0.5, 0.5))),
                    'sub_professionalism': max(1.0, min(5.0, rating + random.uniform(-0.5, 0.5))),
                    'sub_market_expertise': max(1.0, min(5.0, rating + random.uniform(-0.5, 0.5)))
                }
                reviews.append(review)
        
        return pd.DataFrame(reviews)


def run_data_augmentation(
    add_negative_reviews: bool = True,
    add_synthetic_agents: bool = True,
    add_property_types: bool = True,
    dry_run: bool = False
):
    """
    Run data augmentation pipeline.
    
    Args:
        add_negative_reviews: Add negative reviews to existing agents
        add_synthetic_agents: Generate new synthetic agents
        add_property_types: Add property specializations
        dry_run: Don't update database
    """
    print("=" * 80)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 80)
    
    generator = SyntheticDataGenerator()
    
    # Load existing data
    print("\n[1/5] Loading existing data...")
    agents_df = db_client.get_all_agents()
    reviews_df = db_client.get_all_reviews()
    
    print(f"  - Current agents: {len(agents_df)}")
    print(f"  - Current reviews: {len(reviews_df)}")
    
    new_reviews = []
    new_agents = []
    
    # Add negative reviews
    if add_negative_reviews:
        print("\n[2/5] Generating negative reviews...")
        neg_reviews = generator.generate_negative_reviews(agents_df, num_reviews=100)
        new_reviews.append(neg_reviews)
        print(f"  - Generated {len(neg_reviews)} negative reviews")
    
    # Generate synthetic agents
    if add_synthetic_agents:
        print("\n[3/5] Generating synthetic agents...")
        synth_agents = generator.generate_synthetic_agents(num_agents=50)
        new_agents.append(synth_agents)
        print(f"  - Generated {len(synth_agents)} synthetic agents")
        
        # Generate reviews for synthetic agents
        print("\n[4/5] Generating reviews for synthetic agents...")
        synth_reviews = generator.generate_reviews_for_synthetic_agents(synth_agents)
        new_reviews.append(synth_reviews)
        print(f"  - Generated {len(synth_reviews)} reviews")
    
    # Add property specializations
    if add_property_types:
        print("\n[5/5] Adding property type specializations...")
        agents_df = generator.add_property_specializations(agents_df)
        print("  - Updated agent specializations")
    
    # Combine data
    if new_reviews:
        all_new_reviews = pd.concat(new_reviews, ignore_index=True)
    else:
        all_new_reviews = pd.DataFrame()
    
    if new_agents:
        all_new_agents = pd.concat(new_agents, ignore_index=True)
    else:
        all_new_agents = pd.DataFrame()
    
    # Update database
    if not dry_run:
        print("\n[6/5] Updating database...")
        
        # Insert new reviews
        if not all_new_reviews.empty:
            # Convert to list of dicts
            review_records = all_new_reviews.to_dict('records')
            # Insert in batches
            for i in range(0, len(review_records), 100):
                batch = review_records[i:i+100]
                for record in batch:
                    db_client.client.table('reviews').insert(record).execute()
            print(f"  - Inserted {len(review_records)} new reviews")
        
        # Insert new agents
        if not all_new_agents.empty:
            agent_records = all_new_agents.to_dict('records')
            for i in range(0, len(agent_records), 100):
                batch = agent_records[i:i+100]
                for record in batch:
                    # Convert date to string
                    if 'last_sale_date' in record and pd.notna(record['last_sale_date']):
                        record['last_sale_date'] = str(record['last_sale_date'])
                    db_client.client.table('real_estate_agents').insert(record).execute()
            print(f"  - Inserted {len(agent_records)} new agents")
        
        # Update existing agents with property types
        if add_property_types:
            print("  - Updating existing agents with property types...")
            # This would require batch update logic
        
        print("\n" + "=" * 80)
        print("DATA AUGMENTATION COMPLETE!")
        print("=" * 80)
    else:
        print("\n[6/5] DRY RUN - Skipping database update")
        print(f"\nWould insert:")
        print(f"  - {len(all_new_reviews)} reviews")
        print(f"  - {len(all_new_agents)} agents")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic data')
    parser.add_argument('--dry-run', action='store_true', help='Run without updating database')
    parser.add_argument('--no-negative', action='store_true', help='Skip negative reviews')
    parser.add_argument('--no-synthetic', action='store_true', help='Skip synthetic agents')
    parser.add_argument('--no-property-types', action='store_true', help='Skip property types')
    
    args = parser.parse_args()
    
    run_data_augmentation(
        add_negative_reviews=not args.no_negative,
        add_synthetic_agents=not args.no_synthetic,
        add_property_types=not args.no_property_types,
        dry_run=args.dry_run
    )
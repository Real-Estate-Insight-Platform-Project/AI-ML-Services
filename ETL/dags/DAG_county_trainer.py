# File: dags/monthly_forecast_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
import pandas as pd
import os
from supabase import create_client

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
DATA_PATH = "/tmp/data3.csv" 
DATA_PATH2 = "/tmp/data4.csv" 

def download_dataset():
    """Download CSV dataset from URL"""
    url = "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_County.csv"
    df = pd.read_csv(url)
    df.to_csv(DATA_PATH, index=False)

def aggregate_data():
    """Load dataset, aggregate, and push to Supabase""" 
    df = pd.read_csv(DATA_PATH)

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    from get_supabase_data import get_supabase_data
    county_lookup = get_supabase_data("county_lookup") 
    state_lookup = get_supabase_data("state_lookup") 

    # Preprocess data
    from preprocessing_3 import preprocess_data_3
    df = preprocess_data_3(df, county_lookup, state_lookup)

    existing_data = get_supabase_data("county_market") 

    # Create year_month composite keys for checking
    existing_data['year_month'] = existing_data['year'].astype(str) + '_' + existing_data['month'].astype(str)
    df['year_month'] = df['year'].astype(str) + '_' + df['month'].astype(str)
    
    # Filter out records that already exist in the database
    existing_year_months = set(existing_data['year_month'])
    
    if df['year_month'].isin(existing_year_months).all():
        print("All records already exist in the database. No new data to insert.")
        raise AirflowSkipException("Skipping remaining tasks as no new data was found")
    else:
        df = df.drop('year_month', axis=1)
        existing_data = existing_data.drop('year_month', axis=1)
        new_data = pd.concat([existing_data, df]).drop_duplicates().reset_index(drop=True)

        data_to_insert = df.to_dict('records')
        supabase.table("county_market").insert(data_to_insert).execute()

        print(f"{len(data_to_insert)} new records pushed to Supabase")

    # Save the aggregated data back to CSV for the next step
    new_data.to_csv(DATA_PATH, index=False)

def train_model():
    """Load dataset, train model, save predictions to Supabase"""
    df = pd.read_csv(DATA_PATH)

    # List of features to predict
    features = [
        "median_listing_price",
        "average_listing_price",
        "median_listing_price_per_square_foot",
        "total_listing_count",
        "median_days_on_market"
    ]

    from preprocessing_4 import preprocess_data_4
    from model_trainer_2 import get_predictions

    target_df = preprocess_data_4(df.copy())
    prediction_df = target_df.copy()

    for feature in features:
        predictions = get_predictions(df, feature)
        prediction_df[feature] = predictions

    # Add market_trend column based on average_listing_price trend
    prediction_df['market_trend'] = 'stable'

    for county_num in prediction_df['county_num'].unique():
        county_data = prediction_df[prediction_df['county_num'] == county_num]

        # Check if we have at least 3 months of predictions to analyze trend
        if len(county_data) >= 3:
            # Get the average_listing_price for the next 3 months
            prices = county_data['average_listing_price'].iloc[:3].values

            # Calculate if trend is rising, declining or stable
            if prices[2] > prices[0]:
                prediction_df.loc[prediction_df['county_num'] == county_num, 'market_trend'] = 'rising'
            elif prices[2] < prices[0]:
                prediction_df.loc[prediction_df['county_num'] == county_num, 'market_trend'] = 'declining'

    for col in features:
        if col in prediction_df.columns:
            prediction_df[col] = prediction_df[col].astype(int)

    # Concatenate all predictions
    data_to_insert = prediction_df.to_dict('records')

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Delete all existing rows in the table before inserting new predictions
    supabase.table("county_predictions").delete().gt("year", 0).execute()
    supabase.table("county_predictions").insert(data_to_insert).execute()
    print(f"{len(data_to_insert)} predictions pushed to Supabase (table overwritten)")

# --------------- DAG ---------------- #

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0,
}

dag = DAG(
    "monthly_county_trainer",
    default_args=default_args,
    description="Download data, retrain model, push results",
    schedule="@monthly", 
    catchup=False
)


# Tasks
download_task = PythonOperator(
    task_id='download_dataset',
    python_callable=download_dataset,
    dag=dag,
)

aggregate_task = PythonOperator(
    task_id='aggregate_data',
    python_callable=aggregate_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model_and_push',
    python_callable=train_model,
    dag=dag,
)

# DAG pipeline
download_task >> aggregate_task >> train_task

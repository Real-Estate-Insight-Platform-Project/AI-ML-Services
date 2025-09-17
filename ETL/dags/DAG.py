# File: dags/monthly_forecast_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import os
from supabase import create_client

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
DATA_PATH = "/tmp/data1.csv" 
DATA_PATH2 = "/tmp/data2.csv" 

def download_dataset():
    """Download CSV dataset from URL"""
    url = "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_State.csv"
    df = pd.read_csv(url)
    df.to_csv(DATA_PATH, index=False)

def aggregate_data():
    """Load dataset, aggregate, and push to Supabase""" 
    df = pd.read_csv(DATA_PATH)

    # Preprocess data
    from preprocessing_1 import preprocess_data_1
    df = preprocess_data_1(df)

    # Push preprocessed data to Supabase (state_market table)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    response = supabase.table("state_market").select("*", count="exact").execute()
    if response.count == 0:
        data_to_insert = df.to_dict('records')
        supabase.table("state_market").insert(data_to_insert).execute()
    else:
        print("Data already exists in Supabase; skipping insertion.")
    
    # Load full dataset from Supabase
    from get_supabase_data import get_supabase_data
    df = get_supabase_data() 
    
    df.to_csv(DATA_PATH, index=False)

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

    from preprocessing_2 import preprocess_data_2
    from model_trainer import get_predictions

    target_df = preprocess_data_2(df.copy())
    prediction_df = target_df.copy()

    for feature in features:
        predictions = get_predictions(df, feature)
        prediction_df[feature] = predictions

    # Add market_trend column based on average_listing_price trend
    prediction_df['market_trend'] = 'stable'
    
    for state in prediction_df['state'].unique():
        state_data = prediction_df[prediction_df['state'] == state]
        
        # Check if we have at least 3 months of predictions to analyze trend
        if len(state_data) >= 3:
            # Get the average_listing_price for the next 3 months
            prices = state_data['average_listing_price'].iloc[:3].values
            
            # Calculate if trend is rising, declining or stable
            if prices[2] > prices[0]:
                prediction_df.loc[prediction_df['state'] == state, 'market_trend'] = 'rising'
            elif prices[2] < prices[0]:
                prediction_df.loc[prediction_df['state'] == state, 'market_trend'] = 'declining'

    for col in features:
        if col in prediction_df.columns:
            prediction_df[col] = prediction_df[col].astype(int)

    # Concatenate all predictions
    data_to_insert = prediction_df.to_dict('records')

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Delete all existing rows in the table before inserting new predictions
    supabase.table("predictions").delete().neq("state", "").execute()
    supabase.table("predictions").insert(data_to_insert).execute()
    print(f"{len(data_to_insert)} predictions pushed to Supabase (table overwritten)")

# --------------- DAG ---------------- #

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0,
}

dag = DAG(
    "monthly_trainer",
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

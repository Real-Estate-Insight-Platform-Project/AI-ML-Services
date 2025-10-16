# File: dags/monthly_forecast_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
import pandas as pd
import os
import json
from google.cloud import bigquery
from google.oauth2 import service_account
from get_bq_data import get_bq_data
from preprocessing_1 import preprocess_data_1
from upload_bq_data import upload_bq_data
from preprocessing_2 import preprocess_data_2
from model_trainer_1 import get_predictions
from dotenv import load_dotenv

load_dotenv()
# Get credentials JSON from environment variable
credentials_json_str = os.getenv('GOOGLE_CREDENTIALS_JSON')

if credentials_json_str:
    # Parse the JSON string and create credentials
    credentials_dict = json.loads(credentials_json_str)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    client = bigquery.Client(credentials=credentials, project=credentials_dict['project_id'])
else:
    # Fallback: try to use service_keys.json file
    credentials_path = os.path.join(os.path.dirname(__file__), 'service_keys.json')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    client = bigquery.Client()

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

    state_lookup = get_bq_data(client,"state_lookup") 

    # Preprocess data
    df = preprocess_data_1(df,state_lookup)

    existing_data = get_bq_data(client,"state_market") 

    # Create year_month composite keys for checking
    existing_data['year_month'] = existing_data['year'].astype(str) + '_' + existing_data['month'].astype(str)
    df['year_month'] = df['year'].astype(str) + '_' + df['month'].astype(str)
    
    # Filter out records that already exist in the database
    existing_year_months = set(existing_data['year_month'])
    
    if df['year_month'].isin(existing_year_months).all():
        print("All records already exist in the database. No new data to insert.")
        existing_data = existing_data.drop('year_month', axis=1)
        new_data = existing_data.copy()
        # raise AirflowSkipException("Skipping remaining tasks as no new data was found")
    else:
        df = df.drop('year_month', axis=1)
        existing_data = existing_data.drop('year_month', axis=1)
        new_data = pd.concat([existing_data, df]).drop_duplicates().reset_index(drop=True)

        upload_bq_data(client, "state_market", df, "WRITE_APPEND")

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

    target_df = preprocess_data_2(df.copy())
    prediction_df = target_df.copy()

    for feature in features:
        predictions = get_predictions(df, feature)
        prediction_df[feature] = predictions

    # Add market_trend column based on average_listing_price trend
    prediction_df['market_trend'] = 'stable'
    
    for state_num in prediction_df['state_num'].unique():
        state_data = prediction_df[prediction_df['state_num'] == state_num]
        
        # Check if we have at least 3 months of predictions to analyze trend
        if len(state_data) >= 3:
            # Get the average_listing_price for the next 3 months
            prices = state_data['average_listing_price'].iloc[:3].values
            
            # Calculate if trend is rising, declining or stable
            if prices[2] > prices[0]:
                prediction_df.loc[prediction_df['state_num'] == state_num, 'market_trend'] = 'rising'
            elif prices[2] < prices[0]:
                prediction_df.loc[prediction_df['state_num'] == state_num, 'market_trend'] = 'declining'

    for col in features:
        if col in prediction_df.columns:
            prediction_df[col] = prediction_df[col].astype(int)

    upload_bq_data(client, "state_predictions", prediction_df, "WRITE_TRUNCATE")

# --------------- DAG ---------------- #

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0,
}

dag = DAG(
    "monthly_state_trainer",
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

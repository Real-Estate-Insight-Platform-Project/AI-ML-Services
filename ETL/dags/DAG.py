# File: dags/monthly_forecast_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor  # example model
from sklearn.preprocessing import LabelEncoder
import joblib

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

    # Push preprocessed data to Supabase (State_Market table)
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    data_to_insert = df.to_dict('records')
    supabase.table("State_Market").insert(data_to_insert).execute()
    
    # Load full dataset from Supabase
    from get_supabase_data import get_supabase_data
    df = get_supabase_data() 
    
    df.to_csv(DATA_PATH, index=False)

def preprocess_data():
    """Load dataset, preprocess, save to CSV""" 
    df = pd.read_csv(DATA_PATH)

    # Preprocess data
    from preprocessing_2 import preprocess_data_2
    df, target_df = preprocess_data_2(df)

    df.to_csv(DATA_PATH, index=False)
    target_df.to_csv(DATA_PATH2, index=False)

def train_model():
    """Load dataset, train model, save predictions to Supabase"""
    df = pd.read_csv(DATA_PATH)
    target_df = pd.read_csv(DATA_PATH2)

    from model_trainer import get_model_predictions
    predictions = get_model_predictions(df, target_df)

    target_df['PredictedPrice'] = predictions

    # Convert predictions to dict
    data_to_insert = target_df[['month_date_yyyymm', 'state', 'PredictedPrice']].to_dict('records')

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Delete all existing rows in the table before inserting new predictions
    supabase.table("Predictions").delete().neq("state", "").execute()
    supabase.table("Predictions").insert(data_to_insert).execute()
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

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model_and_push',
    python_callable=train_model,
    dag=dag,
)

# DAG pipeline
download_task >> aggregate_task >> preprocess_task >> train_task

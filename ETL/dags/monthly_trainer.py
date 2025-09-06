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
DATA_PATH = "/tmp/RDC_Inventory_Core_Metrics_State.csv"  # change to your path

# --------------- FUNCTIONS ---------------- #

def download_dataset():
    """Download CSV dataset from URL"""
    url = "https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_State.csv"
    df = pd.read_csv(url)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset downloaded to {DATA_PATH}")

def train_model():
    """Load dataset, train model, save predictions to Supabase"""
    df = pd.read_csv(DATA_PATH)
    
    # --- Example preprocessing ---
    # Replace this with your actual feature columns and target
    feature_cols = [col for col in df.columns if col not in ['state', 'month_date_yyyymm', 'median_listing_price']]
    le = LabelEncoder()
    df['state_id'] = le.fit_transform(df['state_id'])
    target_col = 'median_listing_price'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model locally (optional)
    joblib.dump(model, "/tmp/latest_model.pkl")
    
    # Generate predictions
    df['PredictedPrice'] = model.predict(X)
    
    # Push to Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Convert predictions to dict
    data_to_insert = df[['State', 'PredictedPrice']].to_dict('records')
    
    supabase.table("Predictions").insert(data_to_insert).execute()
    print(f"{len(data_to_insert)} predictions pushed to Supabase")

# --------------- DAG ---------------- #

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
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

train_task = PythonOperator(
    task_id='train_model_and_push',
    python_callable=train_model,
    dag=dag,
)

# DAG pipeline
download_task >> train_task

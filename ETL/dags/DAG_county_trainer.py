# File: dags/monthly_forecast_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
from google.cloud import bigquery
from google.oauth2 import service_account
from get_bq_data import get_bq_data
from preprocessing_3 import preprocess_data_3
from upload_bq_data import upload_bq_data
from preprocessing_4 import preprocess_data_4
from model_trainer_2 import get_predictions
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

    county_lookup = get_bq_data(client,"county_lookup") 
    state_lookup = get_bq_data(client,"state_lookup") 

    # Preprocess data
    df = preprocess_data_3(df, county_lookup, state_lookup)

    existing_data = get_bq_data(client,"county_market") 

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

        upload_bq_data(client, "county_market", df, "WRITE_APPEND")

    # Save the aggregated data back to CSV for the next step
    new_data.to_csv(DATA_PATH, index=False)

def train_model():
    """Load dataset, train model, save predictions to Supabase"""
    df = pd.read_csv(DATA_PATH)
    df = df[df['year'] >= 2022] # Focus on data from 2021 onwards
    print(f"unique values: {df['year'].unique()}")

    # # List of features to predict
    # features = [
    #     "median_listing_price",
    #     "average_listing_price",
    #     "median_listing_price_per_square_foot",
    #     "total_listing_count",
    #     "median_days_on_market"
    # ]

    # target_df = preprocess_data_4(4, df.copy())
    # prediction_df = target_df.copy()

    # for feature in features:
    #     predictions = get_predictions(df, feature, 3)
    #     prediction_df[feature] = predictions

    # # Add market_trend column based on average_listing_price trend
    # prediction_df['market_trend'] = 'stable'
    # prediction_df['buyer_friendly'] = 0

    # for county_num in prediction_df['county_num'].unique():
    #     county_data = prediction_df[prediction_df['county_num'] == county_num]

    #     # Check if we have at least 3 months of predictions to analyze trend
    #     if len(county_data) >= 3:
    #         # Get the median_listing_price for the next 3 months
    #         prices = county_data['median_listing_price'].iloc[:3].values

    #         # Get total_listing_count and median_days_on_market for 1st and 3rd month
    #         listing_count_1st = county_data['total_listing_count'].iloc[0]
    #         listing_count_3rd = county_data['total_listing_count'].iloc[2]
    #         days_on_market_1st = county_data['median_days_on_market'].iloc[0]
    #         days_on_market_3rd = county_data['median_days_on_market'].iloc[2]

    #         # Calculate percentage changes between consecutive months
    #         pct_change_1_to_2 = (prices[1] - prices[0]) / prices[0] if prices[0] != 0 else 0
    #         pct_change_2_to_3 = (prices[2] - prices[1]) / prices[1] if prices[1] != 0 else 0

    #         # Sum the percentage changes
    #         total_pct_change = pct_change_1_to_2 + pct_change_2_to_3

    #         # Calculate if trend is rising, declining or stable
    #         if total_pct_change > 0.01:
    #             prediction_df.loc[prediction_df['county_num'] == county_num, 'market_trend'] = 'rising'
    #         elif total_pct_change < -0.01:
    #             prediction_df.loc[prediction_df['county_num'] == county_num, 'market_trend'] = 'declining'

    #             # If market is declining and both inventory and days on market are increasing,
    #             # mark as buyer friendly
    #             if (listing_count_3rd > listing_count_1st or 
    #                 days_on_market_3rd > days_on_market_1st):
    #                 prediction_df.loc[prediction_df['county_num'] == county_num, 'buyer_friendly'] = 1

    # for col in features:
    #     if col in prediction_df.columns:
    #         prediction_df[col] = prediction_df[col].astype(int)

    # upload_bq_data(client, "county_predictions", prediction_df, "WRITE_TRUNCATE")

def get_insights():

    df = pd.read_csv(DATA_PATH)
    df = df[df['year'] >= 2021] # Focus on data from 2021 onwards

    target_df = preprocess_data_4(13, df.copy())
    prediction_df = target_df.copy()

    features = [
        "median_listing_price",
        "median_days_on_market"
    ]
    
    for feature in features:
        predictions = get_predictions(df, feature, 12)
        prediction_df[feature] = predictions

    for col in features:
        if col in prediction_df.columns:
            prediction_df[col] = prediction_df[col].astype(int)

    preds = prediction_df.copy()

    latest_df = df.groupby("county_num").tail(1).reset_index(drop=True)
    preds['year_month'] = (preds['year'] - 2025) * 12 + preds['month'] - 9
    preds.drop(columns=['year', 'month'], inplace=True)
    cols_order = ['year_month', 'county_num', 'state_num', 'median_listing_price', 'median_days_on_market']
    preds = preds[cols_order]
    preds = preds[preds['year_month'].isin([3,6,12])]
    preds.loc[preds['year_month'] == 12, 'median_listing_price'] = preds.loc[preds['year_month'] == 12, 'median_listing_price'] * 1.15
    preds.loc[preds['year_month'] == 6, 'median_listing_price'] = preds.loc[preds['year_month'] == 6, 'median_listing_price'] * 1.1

    preds['appreciation'] = 0.0
    for index, row in preds.iterrows():
        county_num = row['county_num']
        
        current_price = latest_df[(latest_df['county_num'] == county_num)]['median_listing_price'].values[0]
        predicted_price = row['median_listing_price']
        
        appreciation = ((predicted_price - current_price) / current_price) * 100
        preds.loc[index, 'appreciation'] = appreciation

    def calculate_volatility(df):
        volatility = {}
        for county in df['county_num'].unique():
            county_data = df[df['county_num'] == county].sort_values(by=['year', 'month'])
            if len(county_data) >= 12:
                last_12_mm = county_data['median_listing_price_mm'].tail(12).values
                volatility[county] = np.std(last_12_mm) * 100
            else:
                volatility[county] = np.nan  # Not enough data to calculate volatility
        return volatility

    volatility_data = calculate_volatility(df)

    preds['volatility'] = 0.0  # Initialize Volatility column

    for index, row in preds.iterrows():
        county_num = row['county_num']
        if county_num in volatility_data:
            preds.loc[index, 'volatility'] = volatility_data[county_num]
        else:
            preds.loc[index, 'volatility'] = np.nan

    def normalize_days_on_market(preds):
        inv_days = preds.groupby('year_month')['median_days_on_market'].transform(
            lambda s: np.where(s != 0, 1 / s, 1)
        )

        scaler = MinMaxScaler()
        preds = preds.copy()
        preds['liquidity'] = scaler.fit_transform(inv_days.to_frame()) * 100
        return preds

    preds = normalize_days_on_market(preds)
    preds.drop(columns=['median_listing_price', 'median_days_on_market'], inplace=True)
    preds['IOI'] = (preds['appreciation']) + (0.2 * preds['liquidity']) - (0.3 * preds['volatility'])
    preds['IOI'] = preds.groupby('year_month')['IOI'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    preds['IOI'] = preds['IOI'] * 100

    upload_bq_data(client, "county_investment_insights", preds, "WRITE_TRUNCATE")

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

insights_task = PythonOperator(
    task_id='get_investment_insights',
    python_callable=get_insights,
    dag=dag,
)

# DAG pipeline
download_task >> aggregate_task >> train_task >> insights_task

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gc
import warnings
warnings.filterwarnings("ignore")
import math
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import StaticCovariatesTransformer, Scaler, InvertibleMapper
from darts.models import XGBModel
from darts.models import LightGBMModel
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
from pathlib import Path

def reduce_memory_usage(data):
    """Compress DataFrame memory by converting data types"""
    initial_memory = data.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Initial memory: {initial_memory:.2f} MB")
    
    for column in data.columns:
        dtype = data[column].dtype
        
        if dtype != object and dtype.name != 'category':
            min_val = data[column].min()
            max_val = data[column].max()
            
            if 'int' in str(dtype):
                if min_val >= -128 and max_val <= 127:
                    data[column] = data[column].astype(np.int8)
                elif min_val >= -32768 and max_val <= 32767:
                    data[column] = data[column].astype(np.int16)
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    data[column] = data[column].astype(np.int32)
            elif 'float' in str(dtype):
                data[column] = data[column].astype(np.float32)
    
    final_memory = data.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Final memory: {final_memory:.2f} MB")
    print(f"Reduced by {100 * (initial_memory - final_memory) / initial_memory:.1f}%")
    
    return data

def cleanup_resources():
    """Force garbage collection and clear cache"""
    gc.collect()
    if hasattr(gc, 'freeze'):
        gc.freeze()

def add_month(test_futures, n_predict):

    for i, ts in enumerate(test_futures):
        if ts is None:
            continue
        last_date = ts.end_time()
        future_ext = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=n_predict, freq="MS")
        
        extra_covs = pd.DataFrame({
            "year": future_ext.year,
            "month": future_ext.month,
        }, index=future_ext)
        
        extra_covs_ts = TimeSeries.from_dataframe(extra_covs)
        test_futures[i] = ts.append(extra_covs_ts)

    return test_futures

def calculate_metrics(y_true, y_hat):
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)

    rmse = math.sqrt(mean_squared_error(y_true, y_hat))
    rmsle = math.sqrt(((np.log1p(np.maximum(0, y_hat)) - np.log1p(np.maximum(0, y_true)))**2).mean())
    mae = mean_absolute_error(y_true, y_hat)
    mape = np.mean(np.abs((y_true - y_hat) / y_true)) * 100
    r2 = r2_score(y_true, y_hat)

    return rmse, rmsle, mae, mape, r2

def get_project_root():
    return Path().resolve().parent.parent

def get_predictions(df,feature):

    # set mlflow tracking uri
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment(experiment_name="RealEstate_forcasting_2")
    # mlflow.set_tracking_uri(uri=(get_project_root() / 'mlflow' / 'mlruns').as_uri())
    # mlflow.set_experiment(experiment_name="RealEstate_forcasting")

    # Memory optimization: reduce DataFrame memory usage
    df = reduce_memory_usage(df.copy())

    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01')
    # sort
    df = df.sort_values(['county_num','date']).reset_index(drop=True)

    # Create covariate matrices
    grouped_ts = {}

    for county_num, g in df.groupby('county_num'):
        # create a TimeSeries with monthly frequency
        ts = TimeSeries.from_dataframe(
        g,
        time_col="date",
        value_cols=feature,
        freq="MS",
        )
        grouped_ts[county_num] = ts

    past_cov_ts = {}
    future_cov_ts = {}
    past_cov_cols = ['median_listing_price','median_listing_price_mm',
        'median_listing_price_yy', 'active_listing_count',
        'active_listing_count_mm', 'active_listing_count_yy',
        'median_days_on_market', 'median_days_on_market_mm',
        'median_days_on_market_yy', 'new_listing_count', 'new_listing_count_mm',
        'new_listing_count_yy', 'price_increased_count',
        'price_increased_count_mm', 'price_increased_count_yy',
        'price_increased_share', 'price_increased_share_mm',
        'price_increased_share_yy', 'price_reduced_count',
        'price_reduced_count_mm', 'price_reduced_count_yy',
        'price_reduced_share', 'price_reduced_share_mm',
        'price_reduced_share_yy', 'pending_listing_count',
        'pending_listing_count_mm', 'pending_listing_count_yy',
        'median_listing_price_per_square_foot',
        'median_listing_price_per_square_foot_mm',
        'median_listing_price_per_square_foot_yy', 'median_square_feet',
        'median_square_feet_mm', 'median_square_feet_yy',
        'average_listing_price', 'average_listing_price_mm',
        'average_listing_price_yy', 'total_listing_count',
        'total_listing_count_mm', 'total_listing_count_yy', 'pending_ratio',
        'pending_ratio_mm', 'pending_ratio_yy']
    
    # Remove the target feature from past_cov_cols
    if feature in past_cov_cols:
        past_cov_cols.remove(feature)
        
    future_cov_cols = ['month','year']  # calendar features known ahead

    for county_num, g in df.groupby('county_num'):
        # Past covariates as a multivariate TimeSeries
        if all(c in g.columns for c in past_cov_cols):
            past_cov_ts[county_num] = TimeSeries.from_dataframe(g, time_col='date', value_cols=past_cov_cols, freq='MS')
        else:
            past_cov_ts[county_num] = None

        if all(c in g.columns for c in future_cov_cols):
            future_cov_ts[county_num] = TimeSeries.from_dataframe(g, time_col='date', value_cols=future_cov_cols, freq='MS')
        else:
            future_cov_ts[county_num] = None

    # Memory cleanup after processing
    del df
    cleanup_resources()

    pipeline_dict = {}
    ts_transformed = {}

    for county_num in grouped_ts:
        log_transformer = InvertibleMapper(np.log1p, np.expm1)   # log1p for target, invertible
        scaler = Scaler()
        pipe = Pipeline([log_transformer, scaler])
        # fit_transform expects a TimeSeries (or list); we pass the one series
        transformed = pipe.fit_transform(grouped_ts[county_num])
        pipeline_dict[county_num] = pipe
        ts_transformed[county_num] = transformed

    n_predict = 3
    train_series = []
    val_series = []
    train_pasts = []
    train_futures = []
    test_futures = []

    for s in ts_transformed:
        ts = ts_transformed[s]

        train = ts[:-n_predict]
        val = ts[-n_predict:]  # last month
        train_series.append(train)
        val_series.append(val)

        train_pasts.append(past_cov_ts[s][:-n_predict])

        train_futures.append(future_cov_ts[s][:-n_predict])
        test_futures.append(future_cov_ts[s]) 

    test_futures = add_month(test_futures, n_predict)
    test_futures = add_month(test_futures, n_predict)
    test_futures = add_month(test_futures, n_predict)

    # XGBoost model training and validation
    with mlflow.start_run(run_name=f"XGB_Darts_Model_{feature}"):

        # Log model hyperparameters
        mlflow.log_params({
            "lags": 12,
            "lags_past_covariates": list(range(-24, 0)),
            "lags_future_covariates": list(range(1, 2)),
            "output_chunk_length": n_predict,
            "random_state": 42
        })

        xgb_model = XGBModel(
            lags=12,
            lags_past_covariates=list(range(-24, 0)),
            lags_future_covariates=list(range(1, 2)),
            output_chunk_length=n_predict,
            random_state=42
        )

        xgb_model.fit(
            series=train_series,
            past_covariates=train_pasts,
            future_covariates=train_futures,
            verbose=True
        )

        preds = xgb_model.predict(
            n=n_predict,
            series=train_series,
            past_covariates=train_pasts,
            future_covariates=test_futures
        )

        y_true, y_hat = [], []
        for j in range (n_predict):
            for i, sname in enumerate(ts_transformed):
                pred_ts = preds[i][j]
                inv = pipeline_dict[sname].inverse_transform(pred_ts)
                y_hat.append(inv.values()[-1].item())

                true_val = val_series[i][j]
                true_inv = pipeline_dict[sname].inverse_transform(true_val)
                y_true.append(true_inv.values()[-1].item())

            xgb_rmse, xgb_rmsle, xgb_mae, xgb_mape, xgb_r2 = calculate_metrics(y_true, y_hat)

            print(f"Validation RMSE: {xgb_rmse:.4f}, RMSLE: {xgb_rmsle:.4f}, "
                f"MAE: {xgb_mae:.4f}, MAPE: {xgb_mape:.2f}%, R²: {xgb_r2:.4f}")

            mlflow.log_metrics({
                f"RMSE_{j+1}_month_ahead": xgb_rmse,
                f"RMSLE_{j+1}_month_ahead": xgb_rmsle,
                f"MAE_{j+1}_month_ahead": xgb_mae,
                f"MAPE_{j+1}_month_ahead": xgb_mape,
                f"R2_{j+1}_month_ahead": xgb_r2
            })

            y_true, y_hat = [], []

        # Log trained model
        # mlflow.xgboost.log_model(xgb_model.model, artifact_path=f"XGB_Darts_Model_{feature}")

    # End MLflow run
    mlflow.end_run()

    # Memory cleanup
    del xgb_model
    cleanup_resources()

    # LightGBM model training and validation
    with mlflow.start_run(run_name=f"LightGBM_Darts_Model_{feature}"):

        # Log model hyperparameters
        mlflow.log_params({
            "lags": 12,
            "lags_past_covariates": list(range(-24, 0)),
            "lags_future_covariates": list(range(1, 2)),
            "output_chunk_length": n_predict,
            "random_state": 42
        })

        lgbm_model = LightGBMModel(
            lags=12,
            lags_past_covariates=list(range(-24, 0)),
            lags_future_covariates=list(range(1, 2)),
            output_chunk_length=n_predict,
            random_state=42
        )

        lgbm_model.fit(
            series=train_series,
            past_covariates=train_pasts,
            future_covariates=train_futures
        )

        preds = lgbm_model.predict(
            n=n_predict,
            series=train_series,
            past_covariates=train_pasts,
            future_covariates=test_futures
        )

        y_true, y_hat = [], []
        for j in range (n_predict):
            for i, sname in enumerate(ts_transformed):
                pred_ts = preds[i][j]
                inv = pipeline_dict[sname].inverse_transform(pred_ts)
                y_hat.append(inv.values()[-1].item())

                true_val = val_series[i][j]
                true_inv = pipeline_dict[sname].inverse_transform(true_val)
                y_true.append(true_inv.values()[-1].item())

            lgb_rmse, lgb_rmsle, lgb_mae, lgb_mape, lgb_r2 = calculate_metrics(y_true, y_hat)

            print(f"Validation RMSE: {lgb_rmse:.4f}, RMSLE: {lgb_rmsle:.4f}, "
                f"MAE: {lgb_mae:.4f}, MAPE: {lgb_mape:.2f}%, R²: {lgb_r2:.4f}")

            mlflow.log_metrics({
                f"RMSE_{j+1}_month_ahead": lgb_rmse,
                f"RMSLE_{j+1}_month_ahead": lgb_rmsle,
                f"MAE_{j+1}_month_ahead": lgb_mae,
                f"MAPE_{j+1}_month_ahead": lgb_mape,
                f"R2_{j+1}_month_ahead": lgb_r2
            })

            y_true, y_hat = [], []

        # Log trained model
        # mlflow.lightgbm.log_model(lgbm_model.model, artifact_path=f"LightGBM_Darts_Model_{feature}")

    # End MLflow run
    mlflow.end_run()

    # Memory cleanup
    del lgbm_model
    cleanup_resources()

    # Predictions for next month
    train_series = []
    train_pasts = []
    train_futures = []
    test_futures = []

    for s in ts_transformed:
        ts = ts_transformed[s]
        train = ts
        train_series.append(train)
        # same slicing for covariates if present
        if past_cov_ts[s] is not None:
            train_pasts.append(past_cov_ts[s])
        else:
            train_pasts.append(None)
        if future_cov_ts[s] is not None:
            train_futures.append(future_cov_ts[s])
            test_futures.append(future_cov_ts[s])
        else:
            train_futures.append(None)

    test_futures = add_month(test_futures, n_predict)
    test_futures = add_month(test_futures, n_predict)


    lags = 12
    lags_past_covariates = list(range(-24,0))   # previous 24 months of past covariates
    lags_future_covariates = list(range(1, 2)) 
    if (lgb_r2 >= xgb_r2):
        print("Using LightGBM model for predictions")
        model = LightGBMModel(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=n_predict,
            random_state=42
        )
    else:
        print("Using XGBoost model for predictions")
        model = XGBModel(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=n_predict,
            random_state=42
        )

    model.fit(series=train_series, past_covariates=train_pasts, future_covariates=train_futures)


    preds = model.predict(
        n=n_predict,
        series=train_series,
        past_covariates=train_pasts,
        future_covariates=test_futures
    )

    y_hat = []

    for j in range (n_predict):
        for i, sname in enumerate(ts_transformed):
            pred_ts = preds[i][j]
            inv = pipeline_dict[sname].inverse_transform(pred_ts)
            y_hat.append(inv.values()[-1].item())

    y_hat = np.array(y_hat)

    # Final cleanup
    cleanup_resources()

    return y_hat
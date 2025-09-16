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


def get_predictions(df,feature):

    def get_project_root():
        return Path().resolve().parent.parent

    # set mlflow tracking uri
    mlflow.set_tracking_uri(uri=(get_project_root() / 'ETL' / 'dags' / 'mlruns').as_uri())
    experiment = mlflow.set_experiment(experiment_name="RealEstate_forcasting")

    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01')
    # sort
    df = df.sort_values(['state','date']).reset_index(drop=True)

    df = df.drop(columns=['state_id'])

    # Create covariate matrices
    grouped_ts = {}

    for state, g in df.groupby('state'):
        # create a TimeSeries with monthly frequency
        ts = TimeSeries.from_dataframe(
        g,
        time_col="date",
        value_cols=feature,
        freq="MS",
        )
        grouped_ts[state] = ts

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

    for state, g in df.groupby('state'):
        # Past covariates as a multivariate TimeSeries
        if all(c in g.columns for c in past_cov_cols):
            past_cov_ts[state] = TimeSeries.from_dataframe(g, time_col='date', value_cols=past_cov_cols, freq='MS')
        else:
            past_cov_ts[state] = None

        if all(c in g.columns for c in future_cov_cols):
            future_cov_ts[state] = TimeSeries.from_dataframe(g, time_col='date', value_cols=future_cov_cols, freq='MS')
        else:
            future_cov_ts[state] = None

    pipeline_dict = {}
    ts_transformed = {}

    for state in grouped_ts:
        log_transformer = InvertibleMapper(np.log1p, np.expm1)   # log1p for target, invertible
        scaler = Scaler()
        pipe = Pipeline([log_transformer, scaler])
        # fit_transform expects a TimeSeries (or list); we pass the one series
        transformed = pipe.fit_transform(grouped_ts[state])
        pipeline_dict[state] = pipe
        ts_transformed[state] = transformed

    n_predict = 1  # next month
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

    # XGBoost model training and validation
    with mlflow.start_run(run_name=f"XGB_Darts_Model_{feature}"):

        # Log model hyperparameters
        mlflow.log_params({
            "lags": 12,
            "lags_past_covariates": list(range(-24, 0)),
            "lags_future_covariates": list(range(1, n_predict+1)),
            "output_chunk_length": n_predict,
            "random_state": 42
        })

        xgb_model = XGBModel(
            lags=12,
            lags_past_covariates=list(range(-24, 0)),
            lags_future_covariates=list(range(1, n_predict+1)),
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
        for i, sname in enumerate(ts_transformed):
            pred_ts = preds[i]
            inv = pipeline_dict[sname].inverse_transform(pred_ts)
            y_hat.append(inv.values()[-1].item())

            true_val = val_series[i]
            true_inv = pipeline_dict[sname].inverse_transform(true_val)
            y_true.append(true_inv.values()[-1].item())

        y_true = np.array(y_true)
        y_hat = np.array(y_hat)

        xgb_rmse = math.sqrt(mean_squared_error(y_true, y_hat))
        xgb_rmsle = math.sqrt(((np.log1p(np.maximum(0, y_hat)) - np.log1p(np.maximum(0, y_true)))**2).mean())
        xgb_mae = mean_absolute_error(y_true, y_hat)
        xgb_mape = np.mean(np.abs((y_true - y_hat) / y_true)) * 100
        xgb_r2 = r2_score(y_true, y_hat)

        print(f"Validation RMSE: {xgb_rmse:.4f}, RMSLE: {xgb_rmsle:.4f}, "
            f"MAE: {xgb_mae:.4f}, MAPE: {xgb_mape:.2f}%, R²: {xgb_r2:.4f}")

        mlflow.log_metrics({
            "RMSE": xgb_rmse,
            "RMSLE": xgb_rmsle,
            "MAE": xgb_mae,
            "MAPE": xgb_mape,
            "R2": xgb_r2
        })

        # Log trained model
        mlflow.xgboost.log_model(xgb_model.model, artifact_path=f"XGB_Darts_Model_{feature}")

    # End MLflow run
    mlflow.end_run()

    gc.collect()

    # LightGBM model training and validation
    with mlflow.start_run(run_name=f"LightGBM_Darts_Model_{feature}"):

        # Log model hyperparameters
        mlflow.log_params({
            "lags": 12,
            "lags_past_covariates": list(range(-24, 0)),
            "lags_future_covariates": list(range(1, n_predict+1)),
            "output_chunk_length": n_predict,
            "random_state": 42
        })

        lgbm_model = LightGBMModel(
            lags=12,
            lags_past_covariates=list(range(-24, 0)),
            lags_future_covariates=list(range(1, n_predict+1)),
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
        for i, sname in enumerate(ts_transformed):
            pred_ts = preds[i]
            inv = pipeline_dict[sname].inverse_transform(pred_ts)
            y_hat.append(inv.values()[-1].item())

            true_val = val_series[i]
            true_inv = pipeline_dict[sname].inverse_transform(true_val)
            y_true.append(true_inv.values()[-1].item())

        y_true = np.array(y_true)
        y_hat = np.array(y_hat)

        lgb_rmse = math.sqrt(mean_squared_error(y_true, y_hat))
        lgb_rmsle = math.sqrt(((np.log1p(np.maximum(0, y_hat)) - np.log1p(np.maximum(0, y_true)))**2).mean())
        lgb_mae = mean_absolute_error(y_true, y_hat)
        lgb_mape = np.mean(np.abs((y_true - y_hat) / y_true)) * 100
        lgb_r2 = r2_score(y_true, y_hat)

        print(f"Validation RMSE: {lgb_rmse:.4f}, RMSLE: {lgb_rmsle:.4f}, "
            f"MAE: {lgb_mae:.4f}, MAPE: {lgb_mape:.2f}%, R²: {lgb_r2:.4f}")

        mlflow.log_metrics({
            "RMSE": lgb_rmse,
            "RMSLE": lgb_rmsle,
            "MAE": lgb_mae,
            "MAPE": lgb_mape,
            "R2": lgb_r2
        })

        # Log trained model
        mlflow.lightgbm.log_model(lgbm_model.model, artifact_path=f"LightGBM_Darts_Model_{feature}")

    # End MLflow run
    mlflow.end_run()

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
    lags_future_covariates = list(range(1, n_predict+1))  # months ahead 

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

    for i, sname in enumerate(ts_transformed):
        pred_ts = preds[i]
        inv = pipeline_dict[sname].inverse_transform(pred_ts)
        y_hat.append(inv.values()[-1].item())

    y_hat = np.array(y_hat)

    return y_hat
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import gc
import warnings
warnings.filterwarnings("ignore")
import math
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import StaticCovariatesTransformer, Scaler, InvertibleMapper
from collections import defaultdict
from darts.models import XGBModel
from darts.models import LightGBMModel
from darts.models import BlockRNNModel
from darts.models import LinearRegressionModel
from darts.models import RandomForestModel
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
    # mlflow.set_tracking_uri("http://host.docker.internal:5000")
    # mlflow.set_experiment(experiment_name="RealEstate_forcasting")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name="RealEstate_forcasting")

    # Memory optimization: reduce DataFrame memory usage
    df = reduce_memory_usage(df.copy())

    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01')
    # sort
    df = df.sort_values(['state_num','date']).reset_index(drop=True)

    # Create covariate matrices
    grouped_ts = {}

    for state_num, g in df.groupby('state_num'):
        # create a TimeSeries with monthly frequency
        ts = TimeSeries.from_dataframe(
        g,
        time_col="date",
        value_cols=feature,
        freq="MS",
        )
        grouped_ts[state_num] = ts

    past_cov_ts = {}
    future_cov_ts = {}
    past_cov_cols = ['median_listing_price','median_listing_price_mm',
        'median_listing_price_yy', 'active_listing_count',
        'active_listing_count_mm', 'active_listing_count_yy',
        'median_days_on_market', 'median_days_on_market_mm',
        'median_days_on_market_yy', 'new_listing_count', 'new_listing_count_mm',
        'new_listing_count_yy', 'price_increased_count',
        'price_increased_share', 'price_reduced_count',
        'price_reduced_share', 'pending_listing_count',
        'pending_listing_count_mm', 'pending_listing_count_yy',
        'median_listing_price_per_square_foot',
        'median_listing_price_per_square_foot_mm',
        'median_listing_price_per_square_foot_yy',
        'average_listing_price', 'average_listing_price_mm',
        'average_listing_price_yy', 'total_listing_count',
        'total_listing_count_mm', 'total_listing_count_yy']
    
    # Remove the target feature from past_cov_cols
    if feature in past_cov_cols:
        past_cov_cols.remove(feature)
        
    future_cov_cols = ['month','year']  # calendar features known ahead

    for state_num, g in df.groupby('state_num'):
        # Past covariates as a multivariate TimeSeries
        if all(c in g.columns for c in past_cov_cols):
            past_cov_ts[state_num] = TimeSeries.from_dataframe(g, time_col='date', value_cols=past_cov_cols, freq='MS')
        else:
            past_cov_ts[state_num] = None

        if all(c in g.columns for c in future_cov_cols):
            future_cov_ts[state_num] = TimeSeries.from_dataframe(g, time_col='date', value_cols=future_cov_cols, freq='MS')
        else:
            future_cov_ts[state_num] = None

    # Memory cleanup after processing
    del df
    cleanup_resources()

    pipeline_dict = {}
    ts_transformed = {}

    for state_num in grouped_ts:
        log_transformer = InvertibleMapper(np.log1p, np.expm1)   # log1p for target, invertible
        scaler = Scaler()
        pipe = Pipeline([log_transformer, scaler])
        # fit_transform expects a TimeSeries (or list); we pass the one series
        transformed = pipe.fit_transform(grouped_ts[state_num])
        pipeline_dict[state_num] = pipe
        ts_transformed[state_num] = transformed

    n_predict = 3
    train_series = []
    val_series = []
    train_pasts = []
    train_futures = []
    test_futures = []

    meta_predictions = defaultdict(lambda: [[] for _ in range(n_predict)])
    meta_y_true = [[] for _ in range(n_predict)]

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

            if not meta_y_true[j]:
                meta_y_true[j] = list(y_true)
            meta_predictions["XGB"][j] = list(y_hat)
    
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

            if not meta_y_true[j]:
                meta_y_true[j] = list(y_true)
            meta_predictions["LightGBM"][j] = list(y_hat)

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

    # BlockRNN model training and validation
    with mlflow.start_run(run_name=f"LSTM_Darts_Model_{feature}"):

        import torch

        # detect GPU
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            # use one GPU (change devices to -1 or a list to use more)
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": 1, "auto_select_gpus": True}
        else:
            pl_trainer_kwargs = {"accelerator": "cpu"}

        input_chunk_length = 12
        output_chunk_length = n_predict
        model_type = "LSTM"
        hidden_dim = 16
        n_rnn_layers = 2
        dropout_rate = 0.1
        output_chunk_shift = 0
        optimizer_lr = 1e-3
        epochs = 20

        log_params = {
            "model": model_type,
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "hidden_dim": hidden_dim,
            "n_rnn_layers": n_rnn_layers,
            "dropout": dropout_rate,
            "output_chunk_shift": output_chunk_shift,
            "optimizer_lr": optimizer_lr,
            "epochs": epochs,
            "accelerator": pl_trainer_kwargs.get("accelerator"),
            "use_gpu": use_gpu,
        }

        if "devices" in pl_trainer_kwargs:
            log_params["devices"] = pl_trainer_kwargs["devices"]
        if "auto_select_gpus" in pl_trainer_kwargs:
            log_params["auto_select_gpus"] = pl_trainer_kwargs["auto_select_gpus"]

        mlflow.log_params(log_params)

        block_rnn = BlockRNNModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            model=model_type,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=dropout_rate,
            # Optional: output_chunk_shift if you want to leave a gap so the model doesn't peek right up to the target
            output_chunk_shift=output_chunk_shift,
            # Loss function, optimizer, etc.
            optimizer_kwargs={"lr": optimizer_lr},
            # Use GPU if available
            pl_trainer_kwargs=pl_trainer_kwargs
        )

        block_rnn.fit(
            series = train_series, 
            past_covariates = train_pasts, 
            verbose = True,
            epochs = epochs
        )

        preds = block_rnn.predict(
            n = n_predict,
            series = train_series,
            past_covariates = train_pasts
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

            if not meta_y_true[j]:
                meta_y_true[j] = list(y_true)
            meta_predictions["LSTM"][j] = list(y_hat)

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
        # mlflow.block_rnn.log_model(block_rnn.model, artifact_path=f"BlockRNN_Darts_Model_{feature}")

    # End MLflow run
    mlflow.end_run()

    # Memory cleanup
    del block_rnn
    cleanup_resources()

    # Linear Regression model training and validation
    with mlflow.start_run(run_name=f"LinearRegression_Darts_Model_{feature}"):

        # Log model hyperparameters
        mlflow.log_params({
            "lags": 12,
            "lags_past_covariates": list(range(-24, 0)),
            "lags_future_covariates": list(range(1, 2)),
            "output_chunk_length": n_predict
        })

        linear_model = LinearRegressionModel(
            lags=12,
            lags_past_covariates=list(range(-24, 0)),
            lags_future_covariates=list(range(1, 2)),
            output_chunk_length=n_predict
        )

        linear_model.fit(
            series=train_series,
            past_covariates=train_pasts,
            future_covariates=train_futures
        )

        preds = linear_model.predict(
            n=n_predict,
            series=train_series,
            past_covariates=train_pasts,
            future_covariates=test_futures
        )

        y_true, y_hat = [], []
        for j in range(n_predict):
            for i, sname in enumerate(ts_transformed):
                pred_ts = preds[i][j]
                inv = pipeline_dict[sname].inverse_transform(pred_ts)
                y_hat.append(inv.values()[-1].item())

                true_val = val_series[i][j]
                true_inv = pipeline_dict[sname].inverse_transform(true_val)
                y_true.append(true_inv.values()[-1].item())

            if not meta_y_true[j]:
                meta_y_true[j] = list(y_true)
            meta_predictions["linear_regression"][j] = list(y_hat)

            linear_rmse, linear_rmsle, linear_mae, linear_mape, linear_r2 = calculate_metrics(y_true, y_hat)

            print(
                f"Validation RMSE: {linear_rmse:.4f}, RMSLE: {linear_rmsle:.4f}, "
                f"MAE: {linear_mae:.4f}, MAPE: {linear_mape:.2f}%, R²: {linear_r2:.4f}"
            )

            mlflow.log_metrics({
                f"RMSE_{j+1}_month_ahead": linear_rmse,
                f"RMSLE_{j+1}_month_ahead": linear_rmsle,
                f"MAE_{j+1}_month_ahead": linear_mae,
                f"MAPE_{j+1}_month_ahead": linear_mape,
                f"R2_{j+1}_month_ahead": linear_r2
            })

            y_true, y_hat = [], []

        # Log trained model
        # mlflow.linear_regression.log_model(linear_model.model, artifact_path=f"LinearRegression_Darts_Model_{feature}")

    # End MLflow run
    mlflow.end_run()

    # Memory cleanup
    del linear_model
    cleanup_resources()


    # # Random Forest model training and validation
    # with mlflow.start_run(run_name=f"RandomForest_Darts_Model_{feature}"):

    #     rf_lags = 12
    #     rf_lags_past = list(range(-24, 0))
    #     rf_lags_future = list(range(1, 2))
    #     rf_output_chunk_length = n_predict
    #     rf_n_estimators = 200
    #     rf_max_depth = 10
    #     rf_min_samples_split = 2
    #     rf_min_samples_leaf = 1
    #     rf_random_state = 42

    #     mlflow.log_params({
    #         "lags": rf_lags,
    #         "lags_past_covariates": rf_lags_past,
    #         "lags_future_covariates": rf_lags_future,
    #         "output_chunk_length": rf_output_chunk_length,
    #         "n_estimators": rf_n_estimators,
    #         "max_depth": rf_max_depth,
    #         "min_samples_split": rf_min_samples_split,
    #         "min_samples_leaf": rf_min_samples_leaf,
    #         "random_state": rf_random_state,
    #     })

    #     rf_model = RandomForestModel(
    #         lags=rf_lags,
    #         lags_past_covariates=rf_lags_past,
    #         lags_future_covariates=rf_lags_future,
    #         output_chunk_length=rf_output_chunk_length,
    #         n_estimators=rf_n_estimators,
    #         max_depth=rf_max_depth,
    #         min_samples_split=rf_min_samples_split,
    #         min_samples_leaf=rf_min_samples_leaf,
    #         random_state=rf_random_state,
    #     )

    #     rf_model.fit(
    #         series=train_series,
    #         past_covariates=train_pasts,
    #         future_covariates=train_futures
    #     )

    #     preds = rf_model.predict(
    #         n=n_predict,
    #         series=train_series,
    #         past_covariates=train_pasts,
    #         future_covariates=test_futures
    #     )

    #     y_true, y_hat = [], []
    #     for j in range(n_predict):
    #         for i, sname in enumerate(ts_transformed):
    #             pred_ts = preds[i][j]
    #             inv = pipeline_dict[sname].inverse_transform(pred_ts)
    #             y_hat.append(inv.values()[-1].item())

    #             true_val = val_series[i][j]
    #             true_inv = pipeline_dict[sname].inverse_transform(true_val)
    #             y_true.append(true_inv.values()[-1].item())

    #         if not meta_y_true[j]:
    #             meta_y_true[j] = list(y_true)
    #         meta_predictions["random_forest"][j] = list(y_hat)

    #         rf_rmse, rf_rmsle, rf_mae, rf_mape, rf_r2 = calculate_metrics(y_true, y_hat)

    #         print(
    #             f"Validation RMSE: {rf_rmse:.4f}, RMSLE: {rf_rmsle:.4f}, "
    #             f"MAE: {rf_mae:.4f}, MAPE: {rf_mape:.2f}%, R²: {rf_r2:.4f}"
    #         )

    #         mlflow.log_metrics({
    #             f"RMSE_{j+1}_month_ahead": rf_rmse,
    #             f"RMSLE_{j+1}_month_ahead": rf_rmsle,
    #             f"MAE_{j+1}_month_ahead": rf_mae,
    #             f"MAPE_{j+1}_month_ahead": rf_mape,
    #             f"R2_{j+1}_month_ahead": rf_r2
    #         })

    #         y_true, y_hat = [], []

    #     # Log trained model
    #     # mlflow.random_forest.log_model(rf_model.model, artifact_path=f"RandomForest_Darts_Model_{feature}")

    # # End MLflow run
    # mlflow.end_run()

    # # Memory cleanup
    # del rf_model
    # cleanup_resources()


    # # Meta-model stacking using base model predictions
    # available_models = [name for name, preds in meta_predictions.items() if any(len(h) > 0 for h in preds)]

    # if available_models:
    #     with mlflow.start_run(run_name=f"MetaLinearRegression_Darts_Model_{feature}"):
    #         mlflow.log_params({
    #             "meta_model": "LinearRegression",
    #             "meta_base_models": ",".join(sorted(available_models)),
    #             "meta_horizons": n_predict,
    #         })

    #         for horizon_idx in range(n_predict):
    #             y_true_vals = meta_y_true[horizon_idx]
    #             if not y_true_vals:
    #                 continue

    #             horizon_models = [
    #                 model_name
    #                 for model_name in available_models
    #                 if len(meta_predictions[model_name][horizon_idx]) == len(y_true_vals)
    #             ]

    #             if not horizon_models:
    #                 continue

    #             X = np.column_stack([
    #                 meta_predictions[model_name][horizon_idx]
    #                 for model_name in horizon_models
    #             ])

    #             y_array = np.array(y_true_vals)

    #             meta_model = LinearRegression()
    #             meta_model.fit(X, y_array)
    #             meta_preds = meta_model.predict(X)

    #             meta_rmse, meta_rmsle, meta_mae, meta_mape, meta_r2 = calculate_metrics(y_array, meta_preds)

    #             mlflow.log_metrics({
    #                 f"Meta_RMSE_{horizon_idx + 1}_month_ahead": meta_rmse,
    #                 f"Meta_RMSLE_{horizon_idx + 1}_month_ahead": meta_rmsle,
    #                 f"Meta_MAE_{horizon_idx + 1}_month_ahead": meta_mae,
    #                 f"Meta_MAPE_{horizon_idx + 1}_month_ahead": meta_mape,
    #                 f"Meta_R2_{horizon_idx + 1}_month_ahead": meta_r2,
    #             })

    #     mlflow.end_run()
    #     cleanup_resources()
    # else:
    #     print("Meta model skipped: no base model predictions available to stack.")
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import numpy as np
import gc
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")


def get_model_predictions(X, y, df_test, model_func):
    test_preds = np.zeros(len(df_test))
    val_preds = np.zeros(len(X))
    cv = KFold(n_splits=10, shuffle=True, random_state=9)

    # Lists to store fold metrics
    rmse_list = []
    mae_list = []
    r2_list = []

    for fold, (train_ind, valid_ind) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
        X_val, y_val = X.iloc[valid_ind], y.iloc[valid_ind]

        model = model_func()

        # Fit based on model type
        if model_func == lgb_model:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(-1)]
            )
        elif model_func == xgb_model:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=0
            )
        elif model_func == catboost_model:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=[0, 1, 2]
            )
        else:  # RF and ET
            model.fit(X_train, y_train)

        gc.collect()

        y_pred_val = model.predict(X_val)
        y_pred_val = y_pred_val.round()

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        mae = mean_absolute_error(y_val, y_pred_val)
        r2 = r2_score(y_val, y_pred_val)

        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

        print("-" * 60)
        print(f"{model_func.__name__} Fold {fold}")
        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        print("-" * 60)
        
        val_preds[valid_ind] = y_pred_val
        test_preds += model.predict(df_test) / cv.n_splits
        gc.collect()

    # Round test predictions
    test_preds = test_preds.round()

    # Print average metrics across folds
    print("=" * 60)
    print(f"{model_func.__name__} CV Average Metrics:")
    print(f"RMSE: {np.mean(rmse_list):.4f} | MAE: {np.mean(mae_list):.4f} | R2: {np.mean(r2_list):.4f}")
    print("=" * 60)

    return val_preds, test_preds, rmse_list

def lgb_model():
    return lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.1, device='gpu')

def xgb_model():
    return xgb.XGBRegressor(
        n_estimators=1000,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        objective='reg:squarederror',
        random_state=42
    )
    
def catboost_model():
    return CatBoostRegressor(
        iterations=1000,
        loss_function='RMSE',
        random_seed=42,
        task_type='GPU',
        eval_metric='RMSE',
        verbose=0
    )
    
def rf_model():
    return RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

def et_model():
    return ExtraTreesRegressor(
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

def get_predictions(df,target_df):
    df2 = df.copy()
    target_df2 = target_df.copy()

    # Columns to encode
    encode_cols = ["state", "state_id", "Region", "Division"]
    # Initialize encoder
    for col in encode_cols:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col])
        target_df2[col] = le.transform(target_df2[col])

    df2.drop(columns=['state'], inplace=True, errors='ignore')
    target_df2.drop(columns=['state'], inplace=True, errors='ignore')

    label = 'median_listing_price' 
    X = df2.drop(columns=[label], axis=1)  
    y = df2[label]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    scaler = StandardScaler()

    target_df2 = target_df2[X.columns.tolist()]
    X_scaled = scaler.fit_transform(X)
    test_scaled = scaler.transform(target_df2)

    # Optional: convert back to DataFrame
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=target_df2.columns)

    lgb_val_preds, lgb_test_preds, lgb_rmse_list = get_model_predictions(X_scaled, y, test_scaled, lgb_model)

    return lgb_test_preds

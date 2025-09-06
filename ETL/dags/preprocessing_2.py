import pandas as pd
import numpy as np

def preprocess_data_2(df):
    """Preprocess data for modeling"""
    # Sort by state and month
    df = df.sort_values(["state_id", "month_date_yyyymm"])

    # Get latest record per state
    latest_df = df.groupby("state_id").tail(1).reset_index(drop=True)

    # Columns to keep unchanged
    id_cols = ["state", "state_id", "month_date_yyyymm"]

    # Rename all other columns with _prev
    rename_dict = {col: col + "_prev" for col in latest_df.columns if col not in id_cols}
    input_df = latest_df.rename(columns=rename_dict)

    # Ensure it's datetime first (without forcing a strict format)
    input_df["month_date_yyyymm"] = pd.to_datetime(input_df["month_date_yyyymm"])

    # Add one month
    input_df["month_date_yyyymm"] = input_df["month_date_yyyymm"] + pd.DateOffset(months=1)

    # Reorder columns
    id_cols = ["state", "state_id", "month_date_yyyymm"]
    input_df = input_df[id_cols + [c for c in input_df.columns if c not in id_cols]]

    # Columns to always keep
    keep_cols = ["state", "state_id", "month_date_yyyymm", "median_listing_price"]

    # Lag columns
    df_lagged = (
        df.groupby("state_id")
        .shift(1)
        .add_suffix("_prev")
    )

    # Build final dataset
    df_lagged = pd.concat([df[keep_cols], df_lagged], axis=1)

    # Drop earliest month per state (no lag available)
    df_lagged = df_lagged.dropna().reset_index(drop=True)


    df_lagged.drop(columns=['state_prev', 'month_date_yyyymm_prev'], inplace=True)

    df_lagged = df_lagged.sort_values(["state_id", "month_date_yyyymm"])

    # Create lag features for 6 and 12 months
    df_lagged["median_listing_price_lag6"] = (
        df_lagged.groupby("state_id")["median_listing_price"].shift(6)
    )
    df_lagged["median_listing_price_lag12"] = (
        df_lagged.groupby("state_id")["median_listing_price"].shift(12)
    )

    input_df = input_df.sort_values(["state_id", "month_date_yyyymm"])

    # Create lag features for 6 and 12 months
    input_df["median_listing_price_lag6"] = (
        input_df.groupby("state_id")["median_listing_price_prev"].shift(5)
    )
    input_df["median_listing_price_lag12"] = (
        input_df.groupby("state_id")["median_listing_price_prev"].shift(11)
    )
    df_lagged.fillna(-1, inplace=True)
    input_df.fillna(-1, inplace=True)

    df_lagged['month_date_yyyymm'] = pd.to_datetime(df_lagged['month_date_yyyymm'])
    df_lagged['year'] = df_lagged['month_date_yyyymm'].dt.year
    df_lagged['month'] = df_lagged['month_date_yyyymm'].dt.month

    input_df['month_date_yyyymm'] = pd.to_datetime(input_df['month_date_yyyymm'])
    input_df['year'] = input_df['month_date_yyyymm'].dt.year
    input_df['month'] = input_df['month_date_yyyymm'].dt.month

    return df_lagged, input_df
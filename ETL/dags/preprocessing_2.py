import pandas as pd
import numpy as np

def preprocess_data_2(df, feature):
    """Preprocess data for modeling"""
    # Sort by state and month
    df = df.sort_values(["state_id", "year", "month"])

    # Get latest record per state
    latest_df = df.groupby("state_id").tail(1).reset_index(drop=True)

    # Columns to keep unchanged
    id_cols = ["state", "state_id", "year", "month", "Region", "Division"]

    # Rename all other columns with _prev
    rename_dict = {col: col + "_prev" for col in latest_df.columns if col not in id_cols}
    input_df = latest_df.rename(columns=rename_dict)

    # Add one month
    if input_df["month"].iloc[0] == 12:
        input_df["month"] = 1
        input_df["year"] = input_df["year"] + 1
    else:   
        input_df["month"] = input_df["month"] + 1

    # Reorder columns
    id_cols = ["year", "month", "state_id", "state"]
    input_df = input_df[id_cols + [c for c in input_df.columns if c not in id_cols]]

    # Columns to always keep
    keep_cols = ["year", "month", "state_id", "state", "Region", "Division"] + feature

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


    df_lagged.drop(columns=['state_prev', 'year_prev', 'month_prev', "Region_prev", "Division_prev"], inplace=True)

    df_lagged = df_lagged.sort_values(["state_id", "year", "month"])

    # Create lag features for 6 and 12 months
    df_lagged["median_feature_lag6"] = (
        df_lagged.groupby("state_id")[feature].shift(6)
    )
    df_lagged["median_feature_lag12"] = (
        df_lagged.groupby("state_id")[feature].shift(12)
    )

    input_df = input_df.sort_values(["state_id", "year", "month"])

    df_lagged["median_feature_lag5"] = (
        df_lagged.groupby("state_id")[feature].shift(5)
    )
    df_lagged["median_feature_lag11"] = (
        df_lagged.groupby("state_id")[feature].shift(11)
    )

    # Create lag features for 6 and 12 months in input_df using last values from df_lagged
    input_df["median_feature_lag6"] = (
        df_lagged.groupby("state_id")["median_feature_lag5"].last().values
    )
    input_df["median_feature_lag12"] = (
        df_lagged.groupby("state_id")["median_feature_lag11"].last().values
    )

    df_lagged = df_lagged.drop(columns=['median_feature_lag5', 'median_feature_lag11'])

    # Cast selected columns to int (fill NaN with 0 before conversion)
    int_columns = [
        'median_listing_price',
        'active_listing_count',
        'median_days_on_market',
        'new_listing_count',
        'price_increased_count',
        'price_reduced_count',
        'median_listing_price_per_square_foot',
        'median_square_feet',
        'average_listing_price',
        'total_listing_count'
    ]
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df_lagged.fillna(-1, inplace=True)
    input_df.fillna(-1, inplace=True)

    return df_lagged, input_df
import pandas as pd
import numpy as np

def preprocess_data_2(df):
    # Sort by state and month
    df = df.sort_values(["state", "year", "month"])

    # Get latest record per state
    latest_df = df.groupby("state").tail(1).reset_index(drop=True)

    # Rename all other columns with _prev
    input_df = latest_df.copy()

    # Add one month
    if input_df["month"].iloc[0] == 12:
        input_df["month"] = 1
        input_df["year"] = input_df["year"] + 1
    else:   
        input_df["month"] = input_df["month"] + 1

    # Reorder columns
    id_cols = ["year", "month", "state"]
    input_df = input_df[id_cols]

    return input_df
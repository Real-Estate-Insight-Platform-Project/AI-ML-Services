import pandas as pd

def preprocess_data_4(duration, df):
    # Sort by county_num and month
    df = df.sort_values(["county_num", "year", "month"])

    # Get latest record per county_num
    latest_df = df.groupby("county_num").tail(1).reset_index(drop=True)
    
    # Create empty dataframe to store results
    result_df = pd.DataFrame()
    
    # Generate data for next 3 months
    for i in range(1, duration):  # 1, 2, 3 (three months ahead)
        input_df = latest_df.copy()
        
        # Calculate new month and year
        new_month = input_df["month"] + i
        new_year = input_df["year"] + (new_month > 12).astype(int)
        new_month = ((new_month - 1) % 12) + 1  # Handle month overflow
        
        input_df["month"] = new_month
        input_df["year"] = new_year
        
        # Reorder columns
        id_cols = ["year", "month", "county_num", "state_num"]
        input_df = input_df[id_cols]

        # Add to results
        result_df = pd.concat([result_df, input_df], ignore_index=True)
        for col in result_df.columns:
            result_df[col] = result_df[col].astype(int)
    
    return result_df
import pandas as pd

def preprocess_data_1(df,state_lookup):
    """Preprocess the input DataFrame"""
    # Convert month_date_yyyymm to datetime
    df['month_date_yyyymm'] = pd.to_datetime(df['month_date_yyyymm'].astype(str), format='%Y%m')
    
    # Drop unnecessary columns
    df.drop(columns=['quality_flag'], inplace=True, errors='ignore')
    
    # Sort by state_id and month_date_yyyymm
    df = df.sort_values(["state", "month_date_yyyymm"])
    df.reset_index(drop=True, inplace=True)
 
    df['month_date_yyyymm'] = pd.to_datetime(df['month_date_yyyymm'])
    df['year'] = df['month_date_yyyymm'].dt.year
    df['month'] = df['month_date_yyyymm'].dt.month

    df.drop(columns=['month_date_yyyymm','state'], inplace=True, errors='ignore')
    
    df = pd.merge(df, state_lookup[['state_id', 'state_num']], on='state_id', how='left')
    df.drop(columns=['state_id'], inplace=True, errors='ignore')
    df = df[['year', 'month', 'state_num'] + [col for col in df.columns if col not in ['year', 'month', 'state_num']]]

    # Cast selected columns to int (fill NaN with 0 before conversion)
    int_columns = [
        'state_num',
        'median_listing_price',
        'active_listing_count',
        'median_days_on_market',
        'new_listing_count',
        'price_increased_count',
        'price_reduced_count',
        'pending_listing_count',
        'median_listing_price_per_square_foot',
        'median_square_feet',
        'average_listing_price',
        'total_listing_count'
    ]
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df
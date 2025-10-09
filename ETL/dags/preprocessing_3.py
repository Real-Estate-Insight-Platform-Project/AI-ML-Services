import pandas as pd

def preprocess_data_3(df, county_lookup, state_lookup):
    """Preprocess the input DataFrame"""
    # Convert month_date_yyyymm to datetime
    df['month_date_yyyymm'] = pd.to_datetime(df['month_date_yyyymm'].astype(str), format='%Y%m')
    df['state_id'] = df['county_name'].str.split(', ').str[1]
    df['state_id'] = df['state_id'].str.upper()
    df['county_name'] = df['county_name'].str.split(',').str[0]
    df = pd.merge(df, state_lookup[['state_id', 'state']].drop_duplicates(), on='state_id', how='left')
    
    # Drop unnecessary columns
    df.drop(columns=['quality_flag'], inplace=True, errors='ignore')
    
    # Sort by state_id and month_date_yyyymm
    df = df.sort_values(["state", "month_date_yyyymm"])
    df.reset_index(drop=True, inplace=True)
 
    df['month_date_yyyymm'] = pd.to_datetime(df['month_date_yyyymm'])
    df['year'] = df['month_date_yyyymm'].dt.year
    df['month'] = df['month_date_yyyymm'].dt.month

    df.drop(columns=['month_date_yyyymm'], inplace=True, errors='ignore')

    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')

    df = pd.merge(df, state_lookup[['state_id', 'state_num']], on='state_id', how='left')
    df.drop(columns=['state_id','state'], inplace=True, errors='ignore')

    df = pd.merge(df, county_lookup[['county_fips', 'county_num']], on='county_fips', how='left')
    df.drop(columns=['county_fips', 'county_name'], inplace=True, errors='ignore')
    
    df = df.sort_values(["year","month","county_num"])
    df = df[['year', 'month', 'county_num', 'state_num'] + [col for col in df.columns if col not in ['year', 'month', 'county_num', 'state_num']]]

    # Cast selected columns to int (fill NaN with 0 before conversion)
    int_columns = [
        'county_num',
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
            df[col] = df[col].astype('int32')

    return df
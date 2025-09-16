import pandas as pd

def preprocess_data_1(df):
    """Preprocess the input DataFrame"""
    # Convert month_date_yyyymm to datetime
    df['month_date_yyyymm'] = pd.to_datetime(df['month_date_yyyymm'].astype(str), format='%Y%m')
    
    # Drop unnecessary columns
    df.drop(columns=['quality_flag'], inplace=True, errors='ignore')
    
    # Sort by state_id and month_date_yyyymm
    df = df.sort_values(["state", "month_date_yyyymm"])
    df.reset_index(drop=True, inplace=True)
    
    # Mapping dictionary: state -> (Region, Division)
    state_to_region_division = {
        # Northeast - New England
        "Connecticut": ("Northeast", "New England"),
        "Maine": ("Northeast", "New England"),
        "Massachusetts": ("Northeast", "New England"),
        "New Hampshire": ("Northeast", "New England"),
        "Rhode Island": ("Northeast", "New England"),
        "Vermont": ("Northeast", "New England"),

        # Northeast - Middle Atlantic
        "New Jersey": ("Northeast", "Middle Atlantic"),
        "New York": ("Northeast", "Middle Atlantic"),
        "Pennsylvania": ("Northeast", "Middle Atlantic"),

        # Midwest - East North Central
        "Illinois": ("Midwest", "East North Central"),
        "Indiana": ("Midwest", "East North Central"),
        "Michigan": ("Midwest", "East North Central"),
        "Ohio": ("Midwest", "East North Central"),
        "Wisconsin": ("Midwest", "East North Central"),

        # Midwest - West North Central
        "Iowa": ("Midwest", "West North Central"),
        "Kansas": ("Midwest", "West North Central"),
        "Minnesota": ("Midwest", "West North Central"),
        "Missouri": ("Midwest", "West North Central"),
        "Nebraska": ("Midwest", "West North Central"),
        "North Dakota": ("Midwest", "West North Central"),
        "South Dakota": ("Midwest", "West North Central"),

        # South - South Atlantic
        "Delaware": ("South", "South Atlantic"),
        "Florida": ("South", "South Atlantic"),
        "Georgia": ("South", "South Atlantic"),
        "Maryland": ("South", "South Atlantic"),
        "North Carolina": ("South", "South Atlantic"),
        "South Carolina": ("South", "South Atlantic"),
        "Virginia": ("South", "South Atlantic"),
        "West Virginia": ("South", "South Atlantic"),
        "District of Columbia": ("South", "South Atlantic"),

        # South - East South Central
        "Alabama": ("South", "East South Central"),
        "Kentucky": ("South", "East South Central"),
        "Mississippi": ("South", "East South Central"),
        "Tennessee": ("South", "East South Central"),

        # South - West South Central
        "Arkansas": ("South", "West South Central"),
        "Louisiana": ("South", "West South Central"),
        "Oklahoma": ("South", "West South Central"),
        "Texas": ("South", "West South Central"),

        # West - Mountain
        "Arizona": ("West", "Mountain"),
        "Colorado": ("West", "Mountain"),
        "Idaho": ("West", "Mountain"),
        "Montana": ("West", "Mountain"),
        "Nevada": ("West", "Mountain"),
        "New Mexico": ("West", "Mountain"),
        "Utah": ("West", "Mountain"),
        "Wyoming": ("West", "Mountain"),

        # West - Pacific
        "Alaska": ("West", "Pacific"),
        "California": ("West", "Pacific"),
        "Hawaii": ("West", "Pacific"),
        "Oregon": ("West", "Pacific"),
        "Washington": ("West", "Pacific"),
    }

    # Create Region and Division columns
    df["Region"] = df["state"].map(lambda x: state_to_region_division.get(x, (None, None))[0])
    df["Division"] = df["state"].map(lambda x: state_to_region_division.get(x, (None, None))[1])

    df['month_date_yyyymm'] = pd.to_datetime(df['month_date_yyyymm'])
    df['year'] = df['month_date_yyyymm'].dt.year
    df['month'] = df['month_date_yyyymm'].dt.month

    df.drop(columns=['month_date_yyyymm'], inplace=True, errors='ignore')

    # Reorder columns
    id_cols = ["year", "month", "state_id", "state"]
    df = df[id_cols + [c for c in df.columns if c not in id_cols]]


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

    return df
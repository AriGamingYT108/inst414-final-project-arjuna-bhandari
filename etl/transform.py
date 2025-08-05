import os
import pandas as pd

def transform_load_data():
    """
    ETL Transform & Load Stage:
    1. Read data/extracted/extracted_data.csv (skip the type header row).
    2. Normalize headers to snake_case (drop suffixes).
    3. Remove the extra header row from the data.
    4. Drop duplicate rows.
    5. Drop fully empty rows.
    5a. Impute any remaining NaNs: numeric→median, categorical→mode.
    6. Clean column names further (remove non-alphanumeric chars).
    7. Convert 'purpose' to category dtype.
    8. Cast known integer columns to int.
    9. Save to data/extracted/transformed_data.csv.
    """
    raw_path = 'data/extracted/extracted_data.csv'
    df = pd.read_csv(raw_path, skiprows=1)

    # Normalize headers
    df.columns = [col.split()[0].strip().lower().replace('.', '_') for col in df.columns]

    # Remove the extra header row
    df = df.iloc[1:].reset_index(drop=True)

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows that are completely empty
    df = df.dropna(how='all')

    # Impute remaining missing values
    num_cols = df.select_dtypes(include=['number']).columns
    for c in num_cols:
        df[c].fillna(df[c].median(), inplace=True)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for c in cat_cols:
        if df[c].isnull().any():
            df[c].fillna(df[c].mode()[0], inplace=True)

    # Further clean column names
    cleaned = []
    for col in df.columns:
        name = ''.join(ch if ch.isalnum() else '_' for ch in col)
        name = '_'.join(filter(None, name.split('_')))
        cleaned.append(name)
    df.columns = cleaned

    # Convert 'purpose' to categorical
    if 'purpose' in df.columns:
        df['purpose'] = df['purpose'].astype('category')

    # Cast known integer cols
    int_cols = ['credit_policy','fico','inq_last_6mths','delinq_2yrs','pub_rec','not_fully_paid']
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # Save
    out_dir = 'data/extracted'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'transformed_data.csv')
    df.to_csv(out_path, index=False)
    print(f"✔ Transformed data saved to {out_path}")

if __name__ == '__main__':
    transform_load_data()
 
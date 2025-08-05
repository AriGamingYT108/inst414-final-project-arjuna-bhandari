# etl/transform_load.py
"""
ETL Transform & Load Stage:
Reads the raw extracted CSV, cleans and normalizes the data, and writes the cleaned file for analysis.

Steps:
1. Load raw data from 'data/extracted/extracted_data.csv', skipping the type-definition row.
2. Normalize column headers to snake_case without suffixes.
3. Remove the redundant header row that remained as data.
4. Drop exact duplicate records.
5. Drop rows where all values are missing.
5a. Impute remaining missing values:
   - Numeric columns: fill NaN with the column median.
   - Categorical columns: fill NaN with the most frequent category.
5b. Drop any columns that are entirely empty (all NaN).
6. Clean column names by removing non-alphanumeric characters and collapsing underscores.
7. Convert the 'purpose' column to a categorical dtype (if present).
8. Cast specified columns to integer dtype for consistency.
9. Save the fully transformed DataFrame to 'data/extracted/transformed_data.csv'.

Usage:
    python etl/transform_load.py
"""
import os
import pandas as pd
import numpy as np


def transform_load_data():
    # Step 1: Load raw data (skip the first row with type info)
    raw_path = 'data/extracted/extracted_data.csv'
    df = pd.read_csv(raw_path, skiprows=1)

    # Step 2: Normalize headers
    # Keep only the base name (drop type suffix), trim whitespace,
    # convert to lowercase, and replace periods with underscores
    df.columns = [col.split()[0].strip().lower().replace('.', '_') for col in df.columns]

    # Step 3: Remove the extraneous header row
    # The first data row duplicates header info, so drop it and reset the index
    df = df.iloc[1:].reset_index(drop=True)

    # Step 4: Drop exact duplicates
    df = df.drop_duplicates()

    # Step 5: Drop rows that are completely empty
    df = df.dropna(how='all')

    # Step 5a: Impute remaining missing values
    # Numeric columns: fill NaNs with the column median
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    # Categorical columns: fill NaNs with the most frequent category
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # Step 5b: Drop any columns that remain entirely empty
    df = df.dropna(axis=1, how='all')

    # Step 6: Further clean column names
    # Replace non-alphanumeric characters with underscores and collapse repeats
    cleaned_cols = []
    for col in df.columns:
        name = ''.join(ch if ch.isalnum() else '_' for ch in col)
        name = '_'.join(filter(None, name.split('_')))
        cleaned_cols.append(name)
    df.columns = cleaned_cols

    # Step 7: Convert 'purpose' to categorical dtype (if present)
    if 'purpose' in df.columns:
        df['purpose'] = df['purpose'].astype('category')

    # Step 8: Cast known integer columns to int dtype
    int_cols = [
        'credit_policy', 'fico', 'inq_last_6mths',
        'delinq_2yrs', 'pub_rec', 'not_fully_paid'
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Step 9: Save the transformed data to CSV for analysis stage
    out_dir = 'data/extracted'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'transformed_data.csv')
    df.to_csv(out_path, index=False)
    print(f"✔ Transformed data saved to {out_path}")


if __name__ == '__main__':
    transform_load_data()

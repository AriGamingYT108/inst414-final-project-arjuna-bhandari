"""
etl/extract.py

Extract raw loan data from the downloaded CSV and store it for later processing.

This module defines:
- extract_data(): Reads the source CSV, performs a quick sanity check,
  and writes the data to 'data/extracted/extracted_data.csv'.

Usage:
    python etl/extract.py
"""
import os
import pandas as pd


def extract_data():
    """
    Load raw loan data from a specified CSV path and save it into the project's
    'data/extracted' directory for downstream ETL steps.

    Steps:
    1. Create 'data/extracted/' if it does not exist.
    2. Read the raw CSV into a pandas DataFrame.
    3. Display the first 5 rows for a sanity check.
    4. Save the DataFrame to 'data/extracted/extracted_data.csv'.
    """
    # 1. Ensure the output directory exists
    out_dir = 'data/extracted'
    os.makedirs(out_dir, exist_ok=True)

    # 2. Load the raw data CSV from the downloads folder
    raw_path = 'C:/Users/swagm/Downloads/final414data.csv'
    df_raw = pd.read_csv(raw_path)

    # 3. Display first few rows to verify correct load
    print("First 5 rows of the raw dataset:")
    print(df_raw.head())

    # 4. Write the extracted data to the project directory
    out_path = os.path.join(out_dir, 'extracted_data.csv')
    df_raw.to_csv(out_path, index=False)
    print(f"✔ Extracted data saved to {out_path}")


if __name__ == '__main__':
    extract_data()

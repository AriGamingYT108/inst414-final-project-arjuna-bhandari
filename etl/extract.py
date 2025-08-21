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
import numpy as np
import pandas as pd
import openml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)



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

    logging.info("Starting data extraction from OpenML...")

    # 1. Ensure the output directory exists
    
    dataset = openml.datasets.get_dataset(43729)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='dataframe',
    target=dataset.default_target_attribute
    )
    df = X.copy()
    if y is not None:
        df['target'] = y

    logging.info("First 5 rows of dataset:\n%s", df.head())
    logging.info("Dataset shape: %s", df.shape)

    df.to_csv("lending_club_api.csv", index=False)

    out_dir = 'data/extracted'
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'lendingclub_apidata.csv')
    df.to_csv(out_path, index=False)
'''
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
'''

if __name__ == '__main__':
    extract_data()

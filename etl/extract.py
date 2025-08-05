import os
import pandas as pd

def extract_data():
    """
    Extracts data from a raw CSV file and saves it into the project's data directory.

    Steps:
    1. Ensure data/extracted/ exists.
    2. Read the raw CSV from the source path.
    3. Print the first few rows for sanity checking.
    4. Save to data/extracted/extracted_data.csv.
    """
    os.makedirs('data/extracted', exist_ok=True)

    raw_path = 'C:/Users/swagm/Downloads/final414data.csv'
    df = pd.read_csv(raw_path)

    print(df.head())

    out_path = 'data/extracted/extracted_data.csv'
    df.to_csv(out_path, index=False)
    print(f"✔ Extracted data saved to {out_path}")

if __name__ == '__main__':
    extract_data()

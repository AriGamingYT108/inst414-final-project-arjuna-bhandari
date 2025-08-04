import os
import pandas as pd





# 1. Load raw data


df = pd.read_csv("data/extracted_data.csv")

# ===== EDA =====
print("# First 5 rows:")
print(df.head())
print("\n# Data Info:")
print(df.info())
print("\n# Missing values per column:")
print(df.isnull().sum())
print("\n# Summary statistics:")
print(df.describe(include='all'))

# ===== Data Cleaning =====

# 2. Drop exact duplicates
df = df.drop_duplicates()

# 3. Handle missing values
#    - Drop rows that are entirely empty
#    - Fill numeric columns with median
#    - Fill categorical/text columns with mode or empty string

df = df.dropna(how='all')

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    modes = df[col].mode()
    fill_val = modes[0] if not modes.empty else ''
    df[col] = df[col].fillna(fill_val)

# 4. Ensure a robust unique ID column
if 'id' not in df.columns:
    df.insert(0, 'id', [str(uuid.uuid4()) for _ in range(len(df))])
else:
    # remove any rows with duplicate ids
    df = df.drop_duplicates(subset=['id'])

# 5. Convert low-cardinality text columns to categorical dtype
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() / len(df) < 0.5:
        df[col] = df[col].astype('category')

# ===== Save Processed Data =====
processed_dir = os.path.join("data", "processed")
os.makedirs(processed_dir, exist_ok=True)
out_path = os.path.join(processed_dir, "processed_data.csv")
df.to_csv(out_path, index=False)
print(f"✔ Processed data saved to {out_path}")

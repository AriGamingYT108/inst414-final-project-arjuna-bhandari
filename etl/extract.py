import os
import pandas as pd

os.makedirs("data/extracted", exist_ok=True)

extracted_data = pd.read_csv("C:/Users/swagm/Downloads/final414data.csv")
print(extracted_data.head())

extracted_data.to_csv('data/extracted/extracted_data.csv', index=False)
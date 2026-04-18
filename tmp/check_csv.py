import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
csv_file = os.path.join(BASE_DIR, "datasets", "legal_dataset_small.csv")

print(f"File exists: {os.path.exists(csv_file)}")
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, usecols=['text'])
    print(f"Total rows in CSV: {len(df)}")
    print(df.head())

import pandas as pd

# Load full dataset
df = pd.read_csv("../datasets/legal_dataset.csv")

print("Original size:", len(df))

# Take only 5000 rows (safe for laptop)
df_small = df.sample(n=2000, random_state=42)

# Save new file
df_small.to_csv("../datasets/legal_dataset_small.csv", index=False)

print("Reduced dataset saved!")
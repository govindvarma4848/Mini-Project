import os
import pandas as pd

judgement_path = "../datasets/IN-Abs/train-data/judgement"
summary_path = "../datasets/IN-Abs/train-data/summary"

data = []

files = os.listdir(judgement_path)

for file in files:
    try:
        with open(os.path.join(judgement_path, file), 'r', encoding='utf-8') as f:
            text = f.read()

        with open(os.path.join(summary_path, file), 'r', encoding='utf-8') as f:
            summary = f.read()

        data.append({"text": text, "summary": summary})
    except Exception as e:
        print(f"Error with {file}: {e}")

df = pd.DataFrame(data)
df.to_csv("../datasets/legal_dataset.csv", index=False)

print("✅ Dataset converted successfully!")
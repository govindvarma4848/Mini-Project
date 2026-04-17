from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset

# Load dataset
df = pd.read_csv("../datasets/legal_dataset_small.csv")
# Remove null values
df = df.dropna()

# Convert everything to string
df["text"] = df["text"].astype(str)
df["summary"] = df["summary"].astype(str)
# Rename columns if needed
df = df.rename(columns={"text": "input_text", "summary": "target_text"})

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Load model
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenization
def preprocess(example):
    inputs = tokenizer(example["input_text"], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(example["target_text"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

dataset = dataset.map(preprocess, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="../model",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train
trainer.train()

# Save model
model.save_pretrained("../model")
tokenizer.save_pretrained("../model")

print("✅ Training completed!")
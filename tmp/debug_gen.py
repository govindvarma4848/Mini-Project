import os
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set up directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_file = os.path.join(BASE_DIR, "datasets", "legal_dataset_small.csv")
model_path = os.path.join(BASE_DIR, "model")

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = "cpu"
model.to(device)

# Load dataset
df = pd.read_csv(dataset_file, usecols=['text'])
all_judgments = df['text'].astype(str).tolist()

# Retrieval
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
judgment_vectors = vectorizer.fit_transform(all_judgments).toarray().astype(np.float32)
index = faiss.IndexFlatL2(judgment_vectors.shape[1])
index.add(judgment_vectors)

def generate_for_query(query):
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    distances, indices = index.search(query_vector, 1)
    retrieved = all_judgments[indices[0][0]]
    
    print(f"\n--- Retrieved Judgment (first 200 chars) ---\n{retrieved[:200]}...")
    
    input_text = f"Summarize the legal situation: {retrieved[:8000]}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=150, 
        repetition_penalty=2.5,
        no_repeat_ngram_size=3,
        num_beams=4
    )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n--- Generated Summary ---\n{summary}")

generate_for_query("a man killed a person")

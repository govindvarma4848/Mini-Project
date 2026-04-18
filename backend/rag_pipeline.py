import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set up directories and file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_file = os.path.join(BASE_DIR, "datasets", "legal_dataset_small.csv")
model_path = os.path.join(BASE_DIR, "model")
embeddings_path = os.path.join(BASE_DIR, "datasets", "embeddings.npy")

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading system on: {device}")

# 1. Load the fine-tuned T5 model and tokenizer
print("🤖 Loading T5 model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.to(device)

# 2. Load the semantic embedder
print("🔄 Loading semantic embedder (MiniLM)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Load and preprocess the dataset
def load_dataset_for_retrieval(csv_file):
    print(f"📂 Loading dataset from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=['text'])
        texts = df['text'].astype(str).tolist()
        print(f"✅ Loaded {len(texts)} legal records.")
        return df, texts
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return pd.DataFrame(), ["No data available."]

df_data, all_texts = load_dataset_for_retrieval(dataset_file)

# 4. Load or generate embeddings
if os.path.exists(embeddings_path):
    print("⚡ Loading saved semantic embeddings...")
    judgment_embeddings = np.load(embeddings_path)
    # Validate embedding size
    if len(judgment_embeddings) != len(all_texts):
        print("⚠️ Embeddings mismatch! Regenerating...")
        judgment_embeddings = embedder.encode(all_texts, show_progress_bar=True)
        np.save(embeddings_path, judgment_embeddings)
else:
    print("⏳ Creating semantic embeddings (first time only)...")
    judgment_embeddings = embedder.encode(all_texts, show_progress_bar=True)
    np.save(embeddings_path, judgment_embeddings)
    print("✅ Embeddings saved!")

def retrieve_judgments(query, top_k=3):
    """
    Retrieve top-k judgments relevant to the query using semantic similarity.
    """
    query_vec = embedder.encode([query])
    scores = cosine_similarity(query_vec, judgment_embeddings).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return [all_texts[i] for i in top_indices]

def pipeline(query, top_k=1, max_length=512):
    """
    RAG pipeline: retrieve relevant legal texts and generate answers.
    """
    retrieved_docs = retrieve_judgments(query, top_k=top_k)
    
    summaries = []
    for doc in retrieved_docs:
        # Prompt style from working local script
        prompt = f"Explain the legal outcome and punishment: {doc.strip()[:8000]}"
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Generation parameters optimized for T5 and legal text
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)
    
    return summaries

if __name__ == "__main__":
    test_query = "A person stole a bike and was caught by police."
    print(f"Testing query: {test_query}")
    results = pipeline(test_query, top_k=1)
    for i, s in enumerate(results):
        print(f"\nResponse {i+1}:\n{s}")
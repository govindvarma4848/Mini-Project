import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration

# =========================
# 1. LOAD DATASET
# =========================
print("📂 Loading dataset...")
df = pd.read_csv("../datasets/legal_dataset_small.csv")

# Clean dataset
df = df.dropna()
df["text"] = df["text"].astype(str)

# =========================
# 2. LOAD TRAINED MODEL
# =========================
print("🤖 Loading trained model...")
model_path = "../model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# =========================
# 3. LOAD / CREATE EMBEDDINGS (FAST OPTIMIZED)
# =========================
print("🔄 Loading/Creating semantic embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

embeddings_path = "../datasets/embeddings.npy"

if os.path.exists(embeddings_path):
    print("⚡ Loading saved embeddings...")
    X = np.load(embeddings_path)
else:
    print("⏳ Creating embeddings (first time only)...")
    X = embedder.encode(df["text"].tolist(), show_progress_bar=True)
    np.save(embeddings_path, X)
    print("✅ Embeddings saved!")

print("✅ RAG system ready!\n")

# =========================
# 4. RETRIEVAL FUNCTION
# =========================
def retrieve(query, top_k=3):
    query_vec = embedder.encode([query])
    scores = cosine_similarity(query_vec, X).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]["text"].tolist()

# =========================
# 5. GENERATION FUNCTION
# =========================
def generate_summary(text):
    prompt = "Explain the legal outcome and punishment: " + text

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=150)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# 6. MAIN LOOP
# =========================
print("⚖️ AI Legal RAG Assistant (type 'exit' to quit)\n")

while True:
    query = input("Enter your legal query: ")

    if query.lower() == "exit":
        print("Exiting...")
        break

    # Step 1: Retrieve relevant cases
    docs = retrieve(query)

    print("\n🔍 Top Relevant Cases:\n")
    for i, doc in enumerate(docs):
        print(f"\n--- Case {i+1} ---\n")
        print(doc[:300])

    # Step 2: Generate answer
    combined_text = " ".join(docs)
    answer = generate_summary(combined_text)

    print("\n🧠 AI Legal Answer:\n")
    print(answer)
    print("\n" + "="*60 + "\n")
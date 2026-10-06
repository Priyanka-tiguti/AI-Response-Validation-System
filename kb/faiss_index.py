import numpy as np
import faiss
import pickle

from chunking import chunks, chunk_truthfulqa
from load_truthfulqa import load_truthfulqa_file
from preprocess import clean_truthfulqa_data


# -----------------------------
# Load TruthfulQA
# -----------------------------

path = r"D:\AI-Response-Validation-System\kb\data\TruthfulQA.csv"

df = load_truthfulqa_file(path)
cleaned_df = clean_truthfulqa_data(df)
truthful_chunks = chunk_truthfulqa(cleaned_df)


# -----------------------------
# Combine reference chunks
# -----------------------------

reference_chunks = chunks + truthful_chunks


# -----------------------------
# Load embeddings
# -----------------------------

squad_embeddings = np.load("../squad_embeddings.npy")
truthfulqa_embeddings = np.load("../truthfulqa_embeddings.npy")

combined_embeddings = np.vstack([
    squad_embeddings,
    truthfulqa_embeddings
]).astype("float32")

print("Combined embeddings shape:", combined_embeddings.shape)


# -----------------------------
# Normalize embeddings
# -----------------------------

faiss.normalize_L2(combined_embeddings)


# -----------------------------
# Create FAISS index
# -----------------------------

dimension = combined_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(combined_embeddings)


# -----------------------------
# Validate
# -----------------------------

assert len(reference_chunks) == index.ntotal

print("Reference chunks:", len(reference_chunks))
print("FAISS vectors:", index.ntotal)
print("Embedding dimension:", dimension)


# -----------------------------
# Save FAISS index
# -----------------------------

faiss.write_index(
    index,
    "reference_index.faiss"
)


# -----------------------------
# Save reference chunks
# -----------------------------

with open("reference_chunks.pkl", "wb") as f:
    pickle.dump(reference_chunks, f)


print("\nFAISS cosine-similarity index rebuilt successfully!")


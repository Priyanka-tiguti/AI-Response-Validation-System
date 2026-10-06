from sentence_transformers import SentenceTransformer
from chunking import chunk_truthfulqa
from load_truthfulqa import load_truthfulqa_file
from preprocess import clean_truthfulqa_data
import numpy as np

# Load TruthfulQA
path = r"D:\AI-Response-Validation-System\kb\data\TruthfulQA.csv"

df = load_truthfulqa_file(path)

# Clean data
cleaned_df = clean_truthfulqa_data(df)

# Create chunks
truthful_chunks = chunk_truthfulqa(cleaned_df)

print("TruthfulQA rows:", len(cleaned_df))
print("TruthfulQA chunks:", len(truthful_chunks))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from chunks
texts = []

for chunk in truthful_chunks:
    texts.append(chunk.page_content)

# Generate embeddings
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Convert to float32
embeddings = embeddings.astype("float32")

# Save embeddings
np.save("truthfulqa_embeddings.npy", embeddings)

#print("TruthfulQA embeddings saved successfully!")
#print("Embedding shape:", embeddings.shape)
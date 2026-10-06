from sentence_transformers import SentenceTransformer
from chunking import chunks
import numpy as np


# Load embedding model

model = SentenceTransformer("all-MiniLM-L6-v2")


# Extract text from SQuAD chunks

texts = []

for chunk in chunks:
    texts.append(chunk.page_content)


print("Number of SQuAD chunks:", len(texts))


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

np.save("squad_embeddings.npy", embeddings)


print("Embeddings saved successfully!")
print("Embedding shape:", embeddings.shape)

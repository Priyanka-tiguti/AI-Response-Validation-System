import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# 2. Load FAISS index
# --------------------------------------------------

index = faiss.read_index(
    str(BASE_DIR / "reference_index.faiss")
)

print("FAISS index loaded successfully.")
print("Number of vectors:", index.ntotal)


# --------------------------------------------------
# 3. Load reference chunks
# --------------------------------------------------

with open(BASE_DIR / "reference_chunks.pkl", "rb") as f:
    reference_chunks = pickle.load(f)

print("Reference chunks loaded successfully.")
print("Number of chunks:", len(reference_chunks))


# --------------------------------------------------
# 4. Load embedding model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")


# --------------------------------------------------
# 5. Retrieval function
# --------------------------------------------------

def retrieve(query, top_k=10):

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype("float32")

    faiss.normalize_L2(query_embedding)

    similarities, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for i, similarity in zip(indices[0], similarities[0]):

        if i == -1:
            continue

        retrieved_chunks.append({
            "chunk": reference_chunks[i],
            "similarity": float(similarity),
            "index": int(i)
        })

    return retrieved_chunks


# --------------------------------------------------
# 6. Test retrieval
# --------------------------------------------------

if __name__ == "__main__":

    query = "What is the capital of India?"

    results = retrieve(
        query,
        top_k=10
    )

    print("\n" + "=" * 70)
    print("QUERY:")
    print(query)
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\nResult {rank}")
        print("-" * 70)

        print("FAISS Index:", result["index"])
        print("Similarity:", result["similarity"])

        print("Retrieved Chunk:")
        print(result["chunk"])

        print("-" * 70)


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
with open("documents.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Simple chunking
documents = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]

# Convert documents into embeddings
embeddings = model.encode(documents)

embeddings = np.array(embeddings).astype("float32")

# Create FAISS vector index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Store embeddings
index.add(embeddings)

print(f"Loaded {len(documents)} document chunks.")

while True:

    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # Convert question into embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Semantic search
    distances, indices = index.search(query_embedding, k=2)

    print("\nMost relevant information:")

    for i in indices[0]:
        print("-", documents[i])
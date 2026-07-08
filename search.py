import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Containerized model pipelines ensure reproducible machine learning deployments.",
    "Retrieval-Augmented Generation relies on high-quality vector embeddings for context.",
    "Geospatial analysis often requires indexing spatial coordinates in a database.",
    "Temporal graphs capture how relationships between entities evolve over time."
]

doc_embeddings = model.encode(documents)

def semantic_search(query: str, top_k: int = 2):
    query_embedding = model.encode(query)
    scores = np.dot(doc_embeddings, query_embedding) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding))
    
    top_results_idx = np.argsort(scores)[::-1][:top_k]
    
    print(f"\nQuery: '{query}'")
    for idx in top_results_idx:
        print(f"Score: {scores[idx]:.4f} | Document: {documents[idx]}")
        
semantic_search("How do I track evolving data relationships?")
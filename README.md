# Local Semantic Search Engine

This is a pure Python implementation of a local semantic search engine. 

It leverages `sentence-transformers` and `NumPy` to encode text into dense vectors and perform high-speed cosine similarity search entirely in memory. This project demonstrates core Information Retrieval (IR) mechanics without relying on external cloud vector databases, heavy orchestration frameworks, or black-box APIs.

## Features
* **Local Inference:** Runs the highly optimized `all-MiniLM-L6-v2` transformer model locally.
* **Mathematical Retrieval:** Computes cosine similarity scores using pure NumPy matrix operations.
* **Zero Cloud Dependency:** Completely self-contained execution environment.

## Quickstart

1. Clone the repository and navigate to the directory:
   ```bash
   git clone <your-repo-url>
   cd local_semantic_search
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install numpy sentence-transformers
   ```

4. Run the search engine:
   ```bash
   python search.py
   ```
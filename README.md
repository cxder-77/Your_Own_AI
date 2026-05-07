# Your-OWN-AI (Python VectorDB)

A complete Python conversion of the original VectorDB demo app.

## What this project includes

- A Python HTTP server using only the standard library
- A static web UI served from index.html
- Demo semantic search over 20 sample vectors
- Three search modes: **HNSW** (simulated), **KD-Tree**, and **Brute Force**
- Document insertion, document search, and a simple RAG-style answer endpoint
- No external Python dependencies required

## Run the app

1. Open a terminal in this folder:
   cd c:\Users\Raghav\OneDrive\Desktop\AI_Project\Your-OWN-AI
2. Start the server:
   python app.py
3. Open your browser and go to:
   http://localhost:8080

## Notes

- The backend is a pure Python implementation and is compatible with Python 3.10+.
- The web UI will keep using the existing index.html file.
- requirements.txt is included for completeness, but there are no external packages required.
- The RAG-style document assistant is implemented locally and does not require Ollama.

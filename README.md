# legalmate
An AI-powered legal assistant that retrieves and explains contract clauses using FAISS + Hugging Face.
# LegalMate – AI Legal Assistant (Hugging Face + FAISS)

⚠️ This project is for demo/education only and **not** a substitute for legal advice.

A minimal retrieval + generation demo:
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Search**: FAISS
- **LLM**: google/flan-t5-(base|large)
- **UI**: Gradio

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py

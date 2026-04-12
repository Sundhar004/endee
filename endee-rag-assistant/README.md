# Endee RAG Assistant

A RAG system built using the Endee vector database.

## Tech Stack
- Vector DB: Endee (https://github.com/endee-io/endee)
- Backend: Python 3.10+, FastAPI
- Embeddings: sentence-transformers
- LLM: Google Flan-T5-small
- Frontend: HTML/CSS/JS
- Container: Docker

## Quick Start
docker-compose up --build

## How Endee Is Used
1. Create Index: 384-dim cosine
2. Upsert Vectors with metadata
3. Similarity Search via query()
4. Filtered search by category

## License
Apache-2.0

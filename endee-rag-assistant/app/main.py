from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from app.services.endee_client import EndeeService
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.utils.chunking import chunk_text

app = FastAPI(title="Endee RAG Assistant", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

endee = EndeeService()
embedder = EmbeddingService()
llm = LLMService()

class DocumentUpload(BaseModel):
    content: str
    document_id: str
    category: str = "general"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
async def root():
    return {"message": "Endee RAG Assistant API", "status": "running"}

@app.post("/api/documents")
async def upload_document(doc: DocumentUpload):
    try:
        try:
            endee.get_index("rag_documents")
        except:
            endee.create_index("rag_documents", 384, "cosine")
        chunks = chunk_text(doc.content, 500, 50)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc.document_id}_chunk_{i}"
            vector = embedder.embed(chunk)
            endee.upsert("rag_documents", [{
                "id": chunk_id,
                "vector": vector,
                "meta": {"text": chunk, "document_id": doc.document_id, "chunk_index": i},
                "filter": {"category": doc.category, "document_id": doc.document_id}
            }])
        return {"status": "success", "document_id": doc.document_id, "chunks_indexed": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_rag(req: QueryRequest):
    try:
        query_vector = embedder.embed(req.query)
        results = endee.search("rag_documents", query_vector, req.top_k, True)
        context = "\n\n".join([r.get("meta", {}).get("text", "") for r in results])
        answer = llm.generate(req.query, context)
        sources = [{"id": r.get("id"), "text": r.get("meta", {}).get("text", "")[:200] + "...", "similarity": round(r.get("similarity", 0), 4)} for r in results]
        return {"answer": answer, "sources": sources, "query_embedding_dim": len(query_vector)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/indexes")
async def list_indexes():
    try:
        return endee.list_indexes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/indexes/{index_name}")
async def delete_index(index_name: str):
    try:
        endee.delete_index(index_name)
        return {"status": "deleted", "index": index_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

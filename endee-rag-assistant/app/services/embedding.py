from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384
    
    def embed(self, text):
        return self.model.encode(text).tolist()
    
    def embed_batch(self, texts):
        return [e.tolist() for e in self.model.encode(texts)]
    
    def get_dimension(self):
        return self.dimension

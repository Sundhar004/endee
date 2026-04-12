from endee import Endee, Precision
import os

class EndeeService:
    def __init__(self):
        self.base_url = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
        self.client = Endee()
        self.client.set_base_url(self.base_url)
    
    def create_index(self, name, dimension=384, space_type="cosine"):
        self.client.create_index(name=name, dimension=dimension, space_type=space_type, precision=Precision.INT8)
        return {"name": name}
    
    def get_index(self, name):
        return self.client.get_index(name=name)
    
    def list_indexes(self):
        return self.client.list_indexes()
    
    def delete_index(self, name):
        self.client.delete_index(name=name)
    
    def upsert(self, index_name, vectors):
        index = self.client.get_index(name=index_name)
        index.upsert(vectors)
        return {"indexed": len(vectors)}
    
    def search(self, index_name, vector, top_k=5, include_meta=True):
        index = self.client.get_index(name=index_name)
        return index.query(vector=vector, top_k=top_k, ef=128, include_vectors=False)

from transformers import pipeline

class LLMService:
    def __init__(self):
        self.generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    
    def generate(self, query, context):
        if not context:
            return "No context found. Please upload documents first."
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        result = self.generator(prompt, max_length=200, num_beams=4)
        return result[0]["generated_text"]

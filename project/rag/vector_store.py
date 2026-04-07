import json
import os
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    
# Initialize sentence transformer if library is available
model = None
if HAS_FAISS:
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Warning: Could not load sentence-transformers model. {e}")
        HAS_FAISS = False

def build_vector_store(data_texts):
    if not HAS_FAISS or not model:
        return None
    
    embeddings = model.encode(data_texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

class DocumentStore:
    def __init__(self, data_file):
        self.texts = []
        self.metadata = []
        self.index = None
        
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    self.texts.append(item.get('content', ''))
                    self.metadata.append(item)
                    
            if self.texts and HAS_FAISS:
                self.index = build_vector_store(self.texts)

    def search(self, query, top_k=2):
        if not HAS_FAISS or not self.index:
            # Mock fallback: return the first top_k documents if FAISS/model is missing
            return self.metadata[:top_k]
            
        query_embedding = model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        
        results = []
        for i in indices[0]:
            if 0 <= i < len(self.metadata):
                results.append(self.metadata[i])
        return results

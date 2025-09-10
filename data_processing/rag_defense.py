import chromadb
from sentence_transformers import SentenceTransformer

class RagDefense:
    def __init__(self, index_path="InjecAgent/data/malicious_index.json", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        import json

        with open(index_path, "r") as f:
            malicious_data = json.load(f)

        self.model = SentenceTransformer(model_name)

        # Persistent storage for reuse
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Reuse or create collection
        self.collection = self.chroma_client.get_or_create_collection(name="malicious_prompts")

        # Add embeddings for malicious dataset
        texts = [d["text"] for d in malicious_data]
        ids = [str(d.get("id", i)) for i, d in enumerate(malicious_data)]
        embeddings = self.model.encode(texts).tolist()

        self.collection.add(documents=texts, embeddings=embeddings, ids=ids)

    def is_malicious(self, text, threshold=0.5):
        query_emb = self.model.encode([text]).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=1)

        distance = results["distances"][0][0]
        return distance < threshold

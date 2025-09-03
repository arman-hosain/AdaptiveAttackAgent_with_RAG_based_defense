import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class RagDefense:
    def __init__(self, index_path="InjecAgent/data/malicious_index.json", model_name="all-MiniLM-L6-v2"):
        # Load malicious prompt list
        with open(index_path, "r") as f:
            self.malicious_data = json.load(f)

        self.model = SentenceTransformer(model_name)

        # Build FAISS index
        embeddings = self.model.encode([d["text"] for d in self.malicious_data], convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def is_malicious(self, text, threshold=0.5):
        """Return True if input is semantically close to known malicious prompts."""
        emb = self.model.encode([text], convert_to_numpy=True)
        D, I = self.index.search(emb, k=1)
        return D[0][0] < threshold

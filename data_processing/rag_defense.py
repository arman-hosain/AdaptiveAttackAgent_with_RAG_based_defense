import json
import chromadb
from sentence_transformers import SentenceTransformer


class RagDefense:
    def __init__(self,
                 index_paths=[
                     "InjecAgent/data/malicious_index.json",
                     "InjecAgent/data/prompt_injection.json",
                     "InjecAgent/data/prompt_injection_multilingual_machine_translated.json"
                 ],
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)

        # Persistent Chroma client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="malicious_prompts"
        )

        texts, ids = [], []
        for path in index_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Each file can contain either a list of dicts or a dict
            if isinstance(data, dict):
                data = [data]

            for i, d in enumerate(data):
                # Use "user_input" if present, else fall back to "text"
                text = d.get("user_input", d.get("text"))
                if not text:
                    continue
                texts.append(text)
                ids.append(f"{path}_{d.get('prompt_id', i)}")

        # Encode and upsert
        embeddings = self.model.encode(texts, normalize_embeddings=True).tolist()
        self.collection.upsert(documents=texts, embeddings=embeddings, ids=ids)

    # def is_malicious(self, text, threshold=0.3):
    #     print(f"checking Text in RAG Defense is : {text}")
    #     query_emb = self.model.encode([text], normalize_embeddings=True).tolist()
    #     results = self.collection.query(query_embeddings=query_emb, n_results=1)
    #
    #     distance = results["distances"][0][0]
    #     print(f"Result: {results}")
    #     decision = distance < threshold
    #     print(f"Decision: {decision}")
    #     return decision

    def search_similar(self, query, k=3):
        query_emb = self.model.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=k)
        docs = results["documents"][0]
        return "\n".join(docs)


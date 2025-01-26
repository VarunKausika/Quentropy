from typing import List
import torch
from transformers import AutoTokenizer, AutoModel

class Embedding:
    def __init__(self, model_name: str = "Alibaba-NLP/gte-large-en-v1.5", max_batch_size: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.MAX_BATCH_SIZE = max_batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i:i + self.MAX_BATCH_SIZE]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Take the mean of the token embeddings (last hidden state) to create the sentence embedding
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
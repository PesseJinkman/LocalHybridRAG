from typing import Optional
from requests.exceptions import RequestException
import ollama
from .caching import LRUCache

class OllamaClient:
    def __init__(self, model: str, cache_capacity: int = 1000):
        self.model = model
        self.client = ollama.Client()
        self.cache = LRUCache(cache_capacity)

    def preload(self) -> None:
        try:
            # warm-up call: empty messages to pull the model into memory
            self.client.chat(model=self.model, messages=[])
        except RequestException:
            pass  # log if desired

    def ask(self, query: str, context: str) -> str:
        key = f"{query}:{context}"
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are a helpful assistant tasked to answer user questions about the company OpenAI."
                        "Use the provided context to answer accurately and comprehensively."
                    )},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nPlease provide a detailed and comprehensive answer:"}
                ]
            )
            out = resp["message"]["content"]
            self.cache.put(key, out)
            return out
        except RequestException as e:
            if "503" in str(e):
                return "The server is currently overloaded. Please try again later."
            return f"An error occurred: {e}"
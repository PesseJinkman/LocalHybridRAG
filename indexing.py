import faiss
import numpy as np


def build_ip_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def search_ip(index, query_vec: np.ndarray, top_k: int = 5):
    D, I = index.search(query_vec.reshape(1, -1), top_k * 2)
    return D[0], I[0]
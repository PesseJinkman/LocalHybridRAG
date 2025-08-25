from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


def get_encoder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    # Ensure 2D shape (n, d)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[0] == 0:
        return mat  # empty (0, d)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def encode_documents(encoder: SentenceTransformer, docs: List[str]) -> np.ndarray:
    # Handle empty early to avoid shape errors
    if not docs:
        dim_getter = getattr(encoder, "get_sentence_embedding_dimension", None)
        if callable(dim_getter):
            dim = int(dim_getter())
        else:
            dim = int(np.asarray(encoder.encode([""], convert_to_tensor=False)).shape[-1])
        return np.empty((0, dim), dtype=np.float32)
    v = encoder.encode(docs, convert_to_tensor=False, show_progress_bar=True, normalize_embeddings=False)
    v = np.asarray(v, dtype=np.float32)
    return _normalize(v)


def encode_query(encoder: SentenceTransformer, q: str) -> np.ndarray:
    v = encoder.encode([q], convert_to_tensor=False, show_progress_bar=False, normalize_embeddings=False)
    v = np.asarray(v, dtype=np.float32)
    return _normalize(v)[0]


def load_or_create_cache(path: str, encoder: SentenceTransformer, docs: List[str]) -> Tuple[np.ndarray, int]:
    import os
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 2 and arr.shape[0] == len(docs):
            return arr, arr.shape[1]
    arr = encode_documents(encoder, docs)
    if arr.shape[0] == 0:
        raise ValueError(
            "No document text was loaded. Ensure your --docs_dir or --files contain readable .txt/.md files with content."
        )
    np.save(path, arr)
    return arr, arr.shape[1]    
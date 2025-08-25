from typing import List, Tuple, Callable
import numpy as np


def harmonic_mean(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    """Element-wise harmonic mean with numerical guards."""
    a = np.clip(a, eps, None)
    b = np.clip(b, eps, None)
    return 2.0 / (1.0 / a + 1.0 / b)


def normalize(x: np.ndarray, eps: float) -> np.ndarray:
    """Min-max normalize to [0,1] with stability for near-constant arrays."""
    mn = float(np.min(x))
    mx = float(np.max(x))
    rng = mx - mn
    if rng < eps:
        return np.zeros_like(x)
    return (x - mn) / (rng + eps)


def fuse_semantic_lexical(
    query: str,
    documents: List[str],   # kept for parity with existing callers
    encoder,
    faiss_index,
    bm25,
    top_k: int,
    eps: float,
    tokenize_fn: Callable[[str], List[str]],
    alpha: float = 0.5,
    candidate_mult: int = 2,
) -> List[Tuple[int, float]]:
    """
    Fuse FAISS semantic scores with BM25 lexical scores using a weighted harmonic mean.

    Args:
        query: user query.
        documents: list of document chunks (not used directly, kept for signature parity).
        encoder: sentence-transformers model (already used to build the FAISS index).
        faiss_index: FAISS IndexFlatIP built on L2-normalized embeddings.
        bm25: BM25Wrapper over tokenized documents.
        top_k: number of final results to return.
        eps: small constant for numerical stability.
        tokenize_fn: function to tokenize the query for BM25.
        alpha: weight in [0,1] for semantic vs lexical contribution.
        candidate_mult: retrieve candidate_mult * top_k from FAISS before fusion.

    Returns:
        List of (doc_index, fused_score) sorted by score desc.
    """
    # ---- Semantic candidates via FAISS (cosine via dot on normalized vecs)
    qv = encoder.encode([query], convert_to_tensor=False, normalize_embeddings=False)
    qv = np.asarray(qv, dtype=np.float32)
    qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)

    k = max(top_k * max(1, int(candidate_mult)), top_k)
    D, I = faiss_index.search(qv, k)
    sem_scores = D[0]
    cand_idx = I[0]

    # ---- Lexical scores for the same candidate set
    tq = tokenize_fn(query)
    bm25_all = bm25.scores(tq)  # shape: [num_docs]
    bm25_scores = bm25_all[cand_idx]

    # ---- Normalize to comparable ranges
    sem_n = normalize(sem_scores, eps)
    bm25_n = normalize(bm25_scores, eps)

    # ---- Weighted harmonic mean
    # H_w(a,b) = 1 / ( w/a + (1-w)/b )
    w = float(np.clip(alpha, 0.0, 1.0))
    a = np.clip(sem_n, eps, None)
    b = np.clip(bm25_n, eps, None)
    fused = 1.0 / (w / a + (1.0 - w) / b)

    order = np.argsort(fused)[::-1][:top_k]
    top_idx = cand_idx[order]
    top_scores = fused[order]
    return [(int(i), float(s)) for i, s in zip(top_idx, top_scores)]
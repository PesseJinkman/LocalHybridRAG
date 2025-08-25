from rank_bm25 import BM25Okapi
from typing import List

class BM25Wrapper:
    def __init__(self, tokenized_corpus: List[List[str]]):
        self._bm25 = BM25Okapi(tokenized_corpus)

    def scores(self, tokenized_query: List[str]):
        import numpy as np
        return np.asarray(self._bm25.get_scores(tokenized_query), dtype=float)
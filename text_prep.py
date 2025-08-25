import re
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class TextPrep:
    def __init__(self, stopwords_lang: str = "english", auto_download: bool = True):
        if auto_download:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        self._stop = set(stopwords.words(stopwords_lang))

    def preprocess_text(self, text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalnum() and t not in self._stop]

    def preprocess_chunk(self, chunk: str) -> str:
        return re.sub(r"\s+", " ", "".join(ch for ch in chunk if ch.isprintable())).strip()

    def split_long_text(self, text: str, max_length: int = 512) -> List[str]:
        words, chunks, current, cur_len = text.split(), [], [], 0
        for w in words:
            add = len(w) + (1 if current else 0)
            if cur_len + add > max_length:
                if current:
                    chunks.append(" ".join(current))
                current, cur_len = [w], len(w)
            else:
                current.append(w)
                cur_len += add
        if current:
            chunks.append(" ".join(current))
        return chunks
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Config:
    # Models
    encoder_model: str = "sentence-transformers/all-mpnet-base-v2"
    ollama_model: str = "deepseek-r1:8b"

    # Chunking & preprocessing
    max_chunk_chars: int = 512
    stopwords_lang: str = "english"

    # Search
    top_k: int = 5
    epsilon: float = 1e-8  # for numeric stability

    # Caching & paths
    embeddings_cache: str = "document_embeddings.npy"

    # Where to load documents from
    docs_dir: Optional[str] = "C:/Users/param/Desktop/Param/ASU MS CS/AI Projects/HybridSearchRAG/data"                 # if set, we will recursively read from this folder
    file_exts: Tuple[str, ...] = (".txt", ".md")  # extensions to include when using docs_dir
    file_paths: List[str] = field(default_factory=list)  # explicit file list alternative

    # Runtime
    nltk_auto_download: bool = True
    typewriter: bool = True          # enable typewriter output
    typing_cps: int = 60  
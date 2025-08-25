# HybridSearchRAG

HybridSearchRAG is a **Retrieval-Augmented Generation (RAG) framework** that combines **lexical search (BM25)** and **semantic embeddings (FAISS)** to deliver more robust and accurate document retrieval for LLM-powered applications. It includes utilities for indexing, caching, hybrid retrieval, and integration with local LLMs such as Ollama.

---

## Features

- **Hybrid Retrieval**: Combines BM25 lexical search (`bm25_lexical.py`) with FAISS embeddings-based semantic similarity (`embeddings.py`).
- **Efficient Indexing**: Tools for preparing and indexing documents (`indexing.py`, `text_prep.py`).
- **Caching Layer**: Speeds up repeated queries with a local cache (`caching.py`).
- **Flexible Configurations**: Centralized configuration management (`config.py`).
- **LLM Integration**: Support for Ollama LLMs (`llm_ollama.py`).
- **Command-Line Interface**: Interact with the system via CLI (`cli.py`, `__main__.py`).
- **Logging**: Structured logging for debugging and monitoring (`logging_setup.py`).
- **Example Data**: Guides and prompt examples under `data/`.

---

## Project Structure

```
HybridSearchRAG/
│── bm25_lexical.py       # BM25 lexical search implementation
│── caching.py            # Query/result caching utilities
│── cli.py                # CLI entry point
│── config.py             # Central configuration
│── embeddings.py         # Embedding generation and similarity search
│── hybrid.py             # Hybrid search logic (BM25 + embeddings)
│── indexing.py           # Indexing pipeline for documents
│── io_utils.py           # File and I/O utilities
│── llm_ollama.py         # Integration with Ollama LLM
│── logging_setup.py      # Logging configuration
│── text_prep.py          # Text preprocessing utilities
│── __main__.py           # Project entry point
│── data/                 # Example guides and documents
│── old_main.py           # Legacy entry point (deprecated)
```

---

## Installation

```bash
git clone https://github.com/yourusername/HybridSearchRAG.git
cd HybridSearchRAG
pip install -r requirements.txt
```

*(If no `requirements.txt` exists, list dependencies manually here, e.g. `rank_bm25`, `sentence-transformers`, `faiss`, `ollama`, etc.)*

---

## Usage

### CLI
Run the project using the CLI:
```bash
python -m HybridSearchRAG --query "How do I use reasoning models in GPT-5?"
```

### As a Module
```python
from HybridSearchRAG.hybrid import HybridRetriever

retriever = HybridRetriever()
results = retriever.search("What’s new in GPT-5?")
for doc in results:
    print(doc)
```

---

## Example Data

The `data/` folder contains sample documents and guides, including:
- `GPT5FrontendGuide.txt`
- `GPT5NewFeatureGuide.txt`
- `GPT5PromptingGuide.txt`
- `GPT5UsingReasoningModels.txt`
- `UsingGPT5.txt`

These can be used to test indexing and retrieval workflows.

---

## Roadmap

- [ ] Add support for vector databases (FAISS, Milvus).
- [ ] Expand LLM backends beyond Ollama.
- [ ] API server for remote retrieval.
- [ ] Evaluation framework for retrieval quality.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---


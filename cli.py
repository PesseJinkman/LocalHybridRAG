import argparse
from typing import List
from .config import Config
from .logging_setup import setup_logging
from .text_prep import TextPrep
from .io_utils import load_documents, discover_text_files
from .embeddings import get_encoder, load_or_create_cache
from .indexing import build_ip_index
from .bm25_lexical import BM25Wrapper
from .llm_ollama import OllamaClient
from .hybrid import fuse_semantic_lexical
import sys
import time
import threading
import itertools

def _type_out(text: str, cps: int = 60) -> None:
    """Print text with a typewriter effect."""
    if cps <= 0:
        print(text)
        return
    delay = 1.0 / float(max(1, cps))
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        # slightly faster on whitespace to keep cadence natural
        time.sleep(delay * (0.5 if ch.isspace() else 1.0))

class _Spinner:
    """Simple console spinner shown while waiting for the model."""
    def __init__(self, label: str = "Thinking"):
        self._label = label
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        frames = itertools.cycle("|/-\\")
        sys.stdout.write(self._label + " ")
        sys.stdout.flush()
        while not self._stop.is_set():
            sys.stdout.write(next(frames))
            sys.stdout.flush()
            time.sleep(0.08)
            sys.stdout.write("\b")
        sys.stdout.write("✔\n")
        sys.stdout.flush()

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)

def build_system(cfg: Config):
    log = setup_logging()
    prep = TextPrep(cfg.stopwords_lang, cfg.nltk_auto_download)

    # Resolve document paths from either --docs_dir or --files
    if cfg.docs_dir:
        log.info(f"Discovering documents under {cfg.docs_dir} ...")
        paths = discover_text_files(cfg.docs_dir, cfg.file_exts)
    else:
        paths = cfg.file_paths

    if not paths:
        raise SystemExit("Please provide either --docs_dir or at least one file via --files ...")

    log.info("Loading and preprocessing documents ...")
    docs = load_documents(paths, prep, cfg.max_chunk_chars)
    if len(docs) < 5:
        log.warning("Not enough document chunks for meaningful search.")

    log.info("Loading encoder and cached embeddings ...")
    enc = get_encoder(cfg.encoder_model)
    embs, dim = load_or_create_cache(cfg.embeddings_cache, enc, docs)

    log.info("Building FAISS index ...")
    index = build_ip_index(embs)

    log.info("Preparing BM25 ...")
    tokens = [prep.preprocess_text(d) for d in docs]
    bm25 = BM25Wrapper(tokens)

    llm = OllamaClient(cfg.ollama_model)
    llm.preload()

    return log, prep, docs, enc, index, bm25, llm


def interactive_loop(cfg: Config):
    log, prep, docs, enc, index, bm25, llm = build_system(cfg)
    log.info("Hybrid search system is ready. Type 'quit' to exit.")

    while True:
        try:
            q = input("Enter your question (or 'quit' to exit): ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.strip().lower() == "quit":
            break

        log.info("Searching ...")
        results = fuse_semantic_lexical(
            query=q,
            documents=docs,
            encoder=enc,
            faiss_index=index,
            bm25=bm25,
            top_k=cfg.top_k,
            eps=cfg.epsilon,
            tokenize_fn=prep.preprocess_text,
        )
        if not results:
            print("No relevant results found. Try another query.")
            continue

        print("\nTop results:\n")
        top_ctx = []
        for rank, (doc_idx, score) in enumerate(results, 1):
            snippet = docs[doc_idx][:200].replace("\n", " ")
            print(f"{rank}. [chunk {doc_idx}] score={score:.4f}  {snippet}...")
            top_ctx.append(docs[doc_idx])

        ctx = "\n".join(top_ctx[:3])
        print("\nGenerating comprehensive answer using Ollama.\n")
        spinner = _Spinner("Talking to Ollama")
        try:
            spinner.start()
            answer = llm.ask(q, ctx)
        finally:
            spinner.stop()
        if cfg.typewriter:
            _type_out(answer, cps=cfg.typing_cps)
            print()
        else:
            print(answer)
        print("\n" + "-" * 60 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid semantic+lexical RAG search with Ollama answer synthesis")
    p.add_argument("--files", nargs="+", help="Input text file(s)", default=None)
    p.add_argument("--docs_dir", help="Folder to recursively read documents from", default=None)
    p.add_argument("--exts", help="Comma-separated list of extensions (used with --docs_dir)", default=".txt,.md")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--encoder_model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--ollama_model", default="deepseek-r1:8b")
    p.add_argument("--max_chunk_chars", type=int, default=512)
    p.add_argument("--embeddings_cache", default="document_embeddings.npy")
    p.add_argument("--typewriter", action="store_true", help="Show a typing animation for answers")
    p.add_argument("--typing_cps", type=int, default=60, help="Typing speed (characters per second)")
    return p


def main():
    args = build_arg_parser().parse_args()

    file_paths = [] if args.files is None else args.files
    exts = tuple(e.strip() for e in args.exts.split(",") if e.strip())

    cfg = Config(
        encoder_model=args.encoder_model,
        ollama_model=args.ollama_model,
        max_chunk_chars=args.max_chunk_chars,
        top_k=args.top_k,
        embeddings_cache=args.embeddings_cache,
        docs_dir=args.docs_dir,
        file_exts=exts,
        file_paths=file_paths,
        typewriter=args.typewriter,
        typing_cps=args.typing_cps,
    )

    interactive_loop(cfg)
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from pathlib import Path
import re
import logging
from .text_prep import TextPrep

log = logging.getLogger("HybridSearchRAG")


def discover_text_files(root: str, exts: Tuple[str, ...] = (".txt", ".md")) -> List[str]:
    p = Path(root)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {root}")
    allow = {e.lower() for e in exts}
    files = [str(f.resolve()) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in allow]
    log.info(f"Discovered {len(files)} files matching {sorted(allow)}")
    if files[:5]:
        log.info("Sample: " + " | ".join(files[:5]))
    return files

def _read_text_with_fallback(p: Path) -> str:
    # Try a few common encodings; last resort: ignore errors
    for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="ignore")

def _process_file(path: str, prep: TextPrep, max_len: int) -> List[str]:
    p = Path(path)
    content = _read_text_with_fallback(p)
    # robust paragraph split: blank lines (handles Windows CRLF too)
    chunks = re.split(r"\r?\n\s*\r?\n+", content.strip())
    out: List[str] = []
    for ch in chunks:
        s = prep.preprocess_chunk(ch)
        # fallback: if preprocessing wipes everything, keep a trimmed original
        if not s:
            s = ch.strip()
        if s:
            out.extend(prep.split_long_text(s, max_len))
    return out

def load_documents(paths: List[str], prep: TextPrep, max_len: int) -> List[str]:
    if not paths:
        return []
    results: List[List[str]] = []
    with ThreadPoolExecutor() as ex:
        futs = [ex.submit(_process_file, p, prep, max_len) for p in paths]
        for f in futs:
            results.append(f.result())
    docs = [c for lst in results for c in lst]
    log.info(f"Prepared {len(docs)} text chunks from {len(paths)} files")
    return docs
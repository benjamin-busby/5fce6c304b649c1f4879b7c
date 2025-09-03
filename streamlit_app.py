#!/usr/bin/env python3
# VECTORISE_CVS.PY - build embeddings + index from cleaned TXT CVs
# Uses FAISS if available; otherwise falls back to NumPy dense search.

from pathlib import Path
import argparse, json, csv, os, time
from typing import List, Dict, Any, Optional, Callable

import numpy as np
from sentence_transformers import SentenceTransformer
from textwrap import shorten

# Try FAISS
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False

HERE           = Path(__file__).resolve().parent
CLEAN_TXT_DIR  = HERE / "Clean Text"
INDEX_DIR      = HERE / "Index"
INDEX_PATH     = INDEX_DIR / "faiss.index"
EMB_NPY        = INDEX_DIR / "embeddings.npy"   # NumPy fallback
META_JSONL     = INDEX_DIR / "metadata.jsonl"
META_CSV       = INDEX_DIR / "metadata.csv"

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS   = 900
CHUNK_OVERLAP = 120
TOPK_DEFAULT  = 10

def _emit(cb: Optional[Callable[[str], None]], msg: str):
    (cb or print)(msg)

def list_txt_files(root: Path):
    if not root.exists():
        return []
    files = list(root.rglob("*.txt")) + list(root.rglob("*.TXT"))
    return sorted({f.resolve() for f in files if f.is_file()})

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def to_words(text: str):
    return text.split()

def chunk_text_words(text: str, max_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = to_words(text)
    if not words:
        return []
    chunks, step, i = [], max(1, max_words - overlap), 0
    while i < len(words):
        ch = " ".join(words[i:i+max_words]).strip()
        if ch:
            chunks.append(ch)
        i += step
    return chunks

def normalise_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def preview(text: str, width=140) -> str:
    return shorten(text.replace("\n", " "), width=width, placeholder="…")

def safe_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

def build_index(
    clean_dir: Path = CLEAN_TXT_DIR,
    index_dir: Path = INDEX_DIR,
    verbose: bool = True,
    callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.index"
    emb_npy    = index_dir / "embeddings.npy"
    meta_jsonl = index_dir / "metadata.jsonl"
    meta_csv   = index_dir / "metadata.csv"

    if not clean_dir.exists():
        _emit(callback, f"[build] Missing input folder: {clean_dir}")
        return {"files": 0, "chunks": 0, "index_path": None, "meta_jsonl": None, "meta_csv": None, "backend": "none"}

    _emit(callback, f"[paths] CLEAN_TXT_DIR = {clean_dir}")
    _emit(callback, f"[paths] INDEX_DIR     = {index_dir}")

    _emit(callback, f"[embed] Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts: List[str] = []
    meta: List[Dict[str, Any]] = []
    per_file_chunks: List[tuple[Path, int, int]] = []

    files = list_txt_files(clean_dir)
    _emit(callback, f"[scan] Found {len(files)} .txt files under {clean_dir}")
    if verbose:
        for p in files:
            _emit(callback, f"       - {p}")

    for path in files:
        raw = read_text(path).strip()
        if not raw:
            per_file_chunks.append((path, 0, 0))
            continue
        chunks = chunk_text_words(raw, CHUNK_WORDS, CHUNK_OVERLAP)
        per_file_chunks.append((path, len(chunks), len(raw)))
        cv_id = path.stem
        for i, ch in enumerate(chunks):
            texts.append(ch)
            meta.append({
                "cv_id": cv_id,
                "chunk_id": i,
                "path": str(path.resolve()),
                "chars": len(ch),
                "words": len(to_words(ch)),
            })

    _emit(callback, "[scan] Per-file chunk counts:")
    for path, cnt, chars in per_file_chunks:
        _emit(callback, f"       {path.name}: chunks={cnt} chars={chars}")

    if not texts:
        _emit(callback, "[build] No text found.")
        return {"files": len(files), "chunks": 0, "index_path": None, "meta_jsonl": None, "meta_csv": None, "backend": "none"}

    total_files = len(files)
    total_chunks = len(texts)
    _emit(callback, f"[build] Files: {total_files} | Chunks: {total_chunks}")

    _emit(callback, "[embed] Encoding chunks...")
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
    X = normalise_rows(X)

    backend = "faiss" if _FAISS_AVAILABLE else "numpy"
    _emit(callback, f"[build] Using backend: {backend}")

    if _FAISS_AVAILABLE:
        dim = X.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(X)
        faiss.write_index(idx, str(index_path))
        index_path_out = str(index_path.resolve())
    else:
        # Save normalized embeddings for NumPy search
        np.save(emb_npy, X)
        index_path_out = str(emb_npy.resolve())

    with meta_jsonl.open("w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cv_id", "chunk_id", "path", "chars", "words"])
        w.writeheader(); w.writerows(meta)

    _emit(callback, f"[save] Metadata JSONL: {meta_jsonl.resolve()}")
    _emit(callback, f"[save] Metadata CSV : {meta_csv.resolve()}")
    _emit(callback, "[done] Index built successfully.")

    return {
        "files": total_files,
        "chunks": total_chunks,
        "index_path": index_path_out,
        "meta_jsonl": str(meta_jsonl.resolve()),
        "meta_csv": str(meta_csv.resolve()),
        "backend": backend,
    }

def load_index_and_meta(index_dir: Path = INDEX_DIR):
    index_path = index_dir / "faiss.index"
    emb_npy    = index_dir / "embeddings.npy"
    meta_jsonl = index_dir / "metadata.jsonl"
    if not meta_jsonl.exists():
        raise FileNotFoundError("Metadata missing - run build_index first.")
    meta = []
    with meta_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                meta.append(json.loads(s))
    if _FAISS_AVAILABLE and index_path.exists():
        return "faiss", faiss.read_index(str(index_path)), meta
    elif emb_npy.exists():
        X = np.load(emb_npy)
        return "numpy", X, meta
    else:
        raise FileNotFoundError("No index found (neither faiss.index nor embeddings.npy).")

def embed_query(model, query: str) -> np.ndarray:
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    return normalise_rows(q)

def search(query: str, topk: int = TOPK_DEFAULT, index_dir: Path = INDEX_DIR, callback=None):
    model = SentenceTransformer(EMBED_MODEL)
    backend, index_obj, meta =

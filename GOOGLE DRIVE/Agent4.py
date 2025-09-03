#!/usr/bin/env python3
# VECTORISE_CVS.PY - build embeddings + FAISS index from cleaned TXT CVs
# Export functions for GUI integration.

from pathlib import Path
import argparse, json, csv, os, time
from typing import List, Dict, Any, Optional, Callable

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from textwrap import shorten

# ----------------------------- PATHS ----------------------------- #
HERE           = Path(__file__).resolve().parent
CLEAN_TXT_DIR  = HERE / "Clean Text"
INDEX_DIR      = HERE / "Index"
INDEX_PATH     = INDEX_DIR / "faiss.index"
META_JSONL     = INDEX_DIR / "metadata.jsonl"
META_CSV       = INDEX_DIR / "metadata.csv"

# ----------------------------- CONFIG ---------------------------- #
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS   = 900
CHUNK_OVERLAP = 120
TOPK_DEFAULT  = 10

# ----------------------------- HELPERS --------------------------- #
def _emit(cb: Optional[Callable[[str], None]], msg: str):
    if cb:
        cb(msg)
    else:
        print(msg)

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

def safe_write_faiss(index: faiss.Index, path: Path, attempts: int = 8, cb=None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.index")
    last_err = None
    for i in range(attempts):
        try:
            faiss.write_index(index, str(tmp))
            os.replace(tmp, path)
            _emit(cb, f"[save] FAISS index -> {path.resolve()}")
            return str(path.resolve())
        except PermissionError as e:
            last_err = e
            time.sleep(0.2 * (2 ** i))
        finally:
            if tmp.exists():
                try: tmp.unlink()
                except Exception: pass
    raise PermissionError(f"Could not update {path} due to lock: {last_err}")

# ------------------------------- BUILD INDEX ------------------------------- #
def build_index(
    clean_dir: Path = CLEAN_TXT_DIR,
    index_dir: Path = INDEX_DIR,
    verbose: bool = True,
    callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.index"
    meta_jsonl = index_dir / "metadata.jsonl"
    meta_csv   = index_dir / "metadata.csv"

    if not clean_dir.exists():
        _emit(callback, f"[build] Missing input folder: {clean_dir}")
        return {"files": 0, "chunks": 0, "index_path": None, "meta_jsonl": None, "meta_csv": None}

    _emit(callback, f"[paths] CLEAN_TXT_DIR = {clean_dir}")
    _emit(callback, f"[paths] INDEX_DIR     = {index_dir}")
    _emit(callback, f"[paths] INDEX_PATH    = {index_path}")

    if index_path.exists():
        try:
            old = faiss.read_index(str(index_path))
            _emit(callback, f"[build] Existing index size: {old.ntotal}")
        except Exception as e:
            _emit(callback, f"[build] Existing index unreadable: {e}")

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
        _emit(callback, "[build] No text found. Check the cleaned folder and file extensions.")
        return {"files": len(files), "chunks": 0, "index_path": None, "meta_jsonl": None, "meta_csv": None}

    total_files = len(files)
    total_chunks = len(texts)
    _emit(callback, f"[build] Files: {total_files} | Chunks: {total_chunks}")

    _emit(callback, "[embed] Encoding chunks...")
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
    X = normalise_rows(X)

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    _emit(callback, f"[build] New index size (ntotal): {index.ntotal}")

    written_index_path = safe_write_faiss(index, index_path, cb=callback)

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
        "index_path": str(Path(written_index_path)),
        "meta_jsonl": str(meta_jsonl.resolve()),
        "meta_csv": str(meta_csv.resolve()),
    }

# ------------------------------- SEARCH ------------------------------- #
def load_index_and_meta(index_dir: Path = INDEX_DIR):
    index_path = index_dir / "faiss.index"
    meta_jsonl = index_dir / "metadata.jsonl"
    if not index_path.exists() or not meta_jsonl.exists():
        raise FileNotFoundError("Index or metadata missing - run build_index first.")
    index = faiss.read_index(str(index_path))
    meta = []
    with meta_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                meta.append(json.loads(s))
    return index, meta

def embed_query(model, query: str) -> np.ndarray:
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    return normalise_rows(q)

def search(query: str, topk: int = TOPK_DEFAULT, index_dir: Path = INDEX_DIR, callback=None):
    model = SentenceTransformer(EMBED_MODEL)
    index, meta = load_index_and_meta(index_dir)
    qv = embed_query(model, query)
    D, I = index.search(qv, topk)
    D, I = D[0], I[0]

    cache = {}
    _emit(callback, f"\n[query] {query}")
    _emit(callback, f"[results] Top {topk}\n" + "-"*80)
    for rank, (idx, score) in enumerate(zip(I, D), start=1):
        m = meta[int(idx)]
        p = Path(m["path"])
        if p not in cache:
            try:
                cache[p] = read_text(p)
            except Exception:
                cache[p] = ""
        snip = preview(cache[p], 220)
        _emit(callback, f"{rank:>2}. score={score:.3f} | {m['cv_id']}#{m['chunk_id']} | {p.name}")
        _emit(callback, f"    {snip}\n")

# ------------------------------- CLI ------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Build and query CV vector index (embeddings + FAISS).")
    ap.add_argument("--build", action="store_true", help="(Re)build the index from cleaned TXT CVs.")
    ap.add_argument("--query", type=str, default=None, help="Run a quick semantic search against the index.")
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT, help="Number of results to show for --query.")
    ap.add_argument("--clean_dir", type=str, default=None, help="Override cleaned TXT directory.")
    ap.add_argument("--index_dir", type=str, default=None, help="Override index directory.")
    return ap.parse_args()

def main():
    args = parse_args()
    clean_dir = Path(args.clean_dir) if args.clean_dir else CLEAN_TXT_DIR
    index_dir = Path(args.index_dir) if args.index_dir else INDEX_DIR
    if args.build:
        build_index(clean_dir=clean_dir, index_dir=index_dir, verbose=True, callback=None)
    if args.query:
        search(args.query, topk=args.topk, index_dir=index_dir, callback=None)
    if not args.build and not args.query:
        print("Use one of:")
        print("  python Agent4.py --build")
        print('  python Agent4.py --query "financial modelling banking insurance"')

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Agent5 - Rank CVs with Azure OpenAI (GUI-friendly)
- Loads FAISS index + metadata
- Builds prompts per CV from top chunks
- Emits progress lines via callback: "X of Y CVs analysed."
- Emits "Complete" when finished
- Returns a summary dict with counts and CSV paths
"""

import os, json, csv, re, time
from pathlib import Path
from collections import defaultdict
from textwrap import shorten
from typing import List, Tuple, Dict, Optional, Callable
from datetime import datetime

import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# ---------- Paths (aligned with Agent4 defaults) ----------
SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_DIR  = SCRIPT_DIR / "Index"          # matches Agent4.INDEX_DIR
INDEX_PATH = INDEX_DIR / "faiss.index"
OUT_CSV    = INDEX_DIR / "ranked.csv"

META_CANDIDATES = [
    INDEX_DIR / "metadata.jsonl",
    INDEX_DIR / "metadata.json",
    INDEX_DIR / "metadata.csv",
    INDEX_DIR / "meta.jsonl",
    INDEX_DIR / "meta.json",
    INDEX_DIR / "meta.csv",
]

# ---------- Retrieval / scoring config ----------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOPN        = 60
PER_CV      = 4

ROLE_BRIEF = (
    "Financial consultant with experience across banking and insurance. "
    "Strong technical abilities in data analysis, SQL, Python, financial modelling, "
    "stakeholder engagement, and delivery in regulated environments. "
    "Culture is where the applicant has a demonstrable number of outside interests "
    "and skills that will enable them to blend into a team environment. "
    "For each CV, create a score for each of the following in one string, using comma-separated numbers: "
    "Domain & Finance Knowledge, Technical & Analytical Skills, Engagement & Communication Skills, "
    "Project Delivery, Leadership, and Culture. "
    "Education background, professional qualifications etc, graduating from a strong university"
)

# ---------- Azure OpenAI ----------
AZURE_OPENAI_ENDPOINT     = os.getenv("AZURE_OPENAI_ENDPOINT",     "https://ai-mon-prod.openai.azure.com/")
AZURE_OPENAI_API_KEY      = os.getenv("AZURE_OPENAI_API_KEY",      "d0f664fdc60047278308c8455bd181a2")
AZURE_OPENAI_API_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION",  "2024-06-01")
AZURE_OPENAI_DEPLOYMENT   = os.getenv("AZURE_OPENAI_DEPLOYMENT",   "gpt-4o-mini")

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 220

# ---------- Metadata loading ----------
_JOBJ_RE = re.compile(r'\{.*\}', re.DOTALL)

def _clean_json_line(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip().lstrip("\ufeff")
    s = "".join(ch for ch in s if ch >= " " or ch in "\t")
    m = _JOBJ_RE.search(s)
    return m.group(0) if m else None

def find_meta_file() -> Path:
    existing = [p for p in META_CANDIDATES if p.exists()]
    if existing:
        return existing[0]
    globs = list(INDEX_DIR.glob("meta*.json*")) + list(INDEX_DIR.glob("metadata*.json*")) + \
            list(INDEX_DIR.glob("meta*.csv")) + list(INDEX_DIR.glob("metadata*.csv"))
    if globs:
        return sorted(globs)[0]
    raise FileNotFoundError(f"No metadata file found in {INDEX_DIR.resolve()}")

def load_meta() -> List[Dict]:
    p = find_meta_file()
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        need = {"cv_id","chunk_id","path"}
        if not need.issubset(cols):
            raise ValueError(f"CSV {p} must include columns: cv_id, chunk_id, path")
        return [
            {"cv_id": r[cols["cv_id"]], "chunk_id": int(r[cols["chunk_id"]]), "path": str(r[cols["path"]])}
            for _, r in df.iterrows()
        ]

    text = p.read_text(encoding="utf-8-sig", errors="ignore").strip()
    if not text:
        raise ValueError(f"{p} is empty.")

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON is not a list.")
        return data

    rows, bad = [], 0
    with p.open("rb") as f:
        for i, bline in enumerate(f, 1):
            s = bline.decode("utf-8-sig", errors="ignore").strip()
            if not s or s.startswith(("#","//")):
                continue
            s = _clean_json_line(s) or ""
            if not s:
                bad += 1
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                bad += 1
                continue
    if not rows:
        raise ValueError(f"No valid records found in {p}. Lines rejected={bad}.")
    return rows

# ---------- Retrieval helpers ----------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def load_index() -> faiss.Index:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH.resolve()}")
    return faiss.read_index(str(INDEX_PATH))

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    q = model.encode([text], convert_to_numpy=True, normalize_embeddings=False)
    return l2_normalize(q)

def group_top_snippets(meta, idxs, dists, per_cv: int):
    from collections import defaultdict
    grouped = defaultdict(list)
    for i, score in zip(idxs, dists):
        m = meta[int(i)]
        grouped[m["cv_id"]].append({
            "chunk_id": m["chunk_id"],
            "score": float(score),
            "path": m["path"],
        })
    for cv_id in list(grouped.keys()):
        grouped[cv_id].sort(key=lambda x: x["score"], reverse=True)
        grouped[cv_id] = grouped[cv_id][:per_cv]
    return grouped

def load_snippet_previews(grouped: dict) -> Dict[str, List[Tuple[str,str]]]:
    cache, out = {}, {}
    for cv_id, items in grouped.items():
        snaps = []
        for it in items:
            p = Path(it["path"])
            if not p.is_absolute():
                p = SCRIPT_DIR / p
            if p not in cache:
                cache[p] = read_text(p)
            preview = shorten(cache[p].replace("\n", " "), width=700, placeholder=" …")
            snaps.append((f"{cv_id}#{it['chunk_id']}", preview))
        out[cv_id] = snaps
    return out

# ---------- Prompt + parsing ----------
def build_prompt(role_brief: str, snippets: List[Tuple[str, str]]) -> Tuple[str, str]:
    ctx = "\n\n".join(f"[{tag}] {txt}" for tag, txt in snippets)
    system = (
        "You are a recruiting assistant. Score strictly from the provided snippets. "
        "Use only evidence in the snippets."
    )
    user = f"""ROLE BRIEF:
{role_brief}

EVIDENCE SNIPPETS:
{ctx}

TASK:
Return exactly two lines:
Line 1: seven integers between 0 and 100, comma-separated, in this order:
1) Domain & Finance Knowledge
2) Technical & Analytical Skills
3) Engagement & Communication Skills
4) Project Delivery
5) Leadership
6) Culture
7) Educational background
Example: 82,76,90,79,70,65,65

Line 2: a compact JSON object with three fields:
{{"summary":"<2-4 sentences>","met":["..."],"missing":["..."]}}

Rules:
- If evidence is missing, be conservative in the score and list it in "missing".
- Do not add any extra lines or text outside these two lines.
"""
    return system, user

_INT_RE = re.compile(r'\d{1,3}')

def parse_seven_scores(line: str) -> List[int]:
    parts = [p.strip() for p in (line or "").split(",")]
    if len(parts) == 7 and all(p.isdigit() for p in parts):
        vals = [int(p) for p in parts]
        if all(0 <= v <= 100 for v in vals):
            return vals
    nums = [int(m.group(0)) for m in _INT_RE.finditer(line or "")]
    nums = [n for n in nums if 0 <= n <= 100]
    if len(nums) >= 7:
        return nums[:7]
    raise ValueError(f"Could not parse seven scores from: {line!r}")

def parse_summary_block(text: str) -> Dict:
    start = text.find("{"); end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end+1])
            return {
                "summary": obj.get("summary", ""),
                "met": obj.get("met", []) or [],
                "missing": obj.get("missing", []) or [],
            }
        except Exception:
            pass
    return {"summary": "", "met": [], "missing": []}

def azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )

def score_with_llm(client: AzureOpenAI, system: str, user: str) -> Tuple[List[int], Dict]:
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    text = (resp.choices[0].message.content or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty response from model.")
    scores = parse_seven_scores(lines[0])
    details = parse_summary_block("\n".join(lines[1:])) if len(lines) > 1 else {"summary":"","met":[],"missing":[]}
    return scores, details

# ---------- Stable CSV writer ----------
def safe_write_csv(df: pd.DataFrame, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.csv")
    last_err = None
    for attempt in range(8):
        try:
            df.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
            os.replace(tmp, path)
            return str(path.resolve())
        except PermissionError as e:
            last_err = e
            time.sleep(0.2 * (2 ** attempt))
        finally:
            if tmp.exists():
                try: tmp.unlink()
                except Exception: pass
    alt = path.with_name(path.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + path.suffix)
    df.to_csv(alt, index=False, quoting=csv.QUOTE_MINIMAL)
    return str(alt.resolve())

# ---------- Public function for GUI ----------
def run_ranking(
    index_dir: Path = INDEX_DIR,
    role_brief: str = ROLE_BRIEF,
    topn: int = TOPN,
    per_cv: int = PER_CV,
    callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """
    Runs the full ranking pipeline and streams progress.
    Emits: "X of Y CVs analysed." after each CV, then "Complete" at the end.

    Returns:
      {"total": Y, "processed": X, "out_csv": "...", "run_csv": "..."}
    """
    # Load resources
    index_path = index_dir / "faiss.index"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path.resolve()}")
    index = faiss.read_index(str(index_path))

    # Metadata
    global INDEX_DIR, INDEX_PATH, OUT_CSV
    INDEX_DIR = index_dir
    INDEX_PATH = index_path
    OUT_CSV = index_dir / "ranked.csv"
    meta = load_meta()

    # Retrieval
    model = SentenceTransformer(EMBED_MODEL)
    qv = embed_query(model, role_brief)
    D, I = index.search(qv, topn)
    d, i = D[0], I[0]
    grouped = group_top_snippets(meta, i, d, per_cv=per_cv)
    snippets_by_cv = load_snippet_previews(grouped)

    # Loop over CVs with progress
    client = azure_client()
    rows = []
    total = len(snippets_by_cv)
    processed = 0

    for processed, (cv_id, snippets) in enumerate(snippets_by_cv.items(), start=1):
        if callback:
            callback(f"{processed} of {total} CVs analysed.")
        system, user = build_prompt(role_brief, snippets)
        scores, details = score_with_llm(client, system, user)
        dom_fin, tech_anal, engage_comm, delivery, leader, culture, education = scores
        rows.append({
            "cv_id": cv_id,
            "domain_finance": dom_fin,
            "technical_analytical": tech_anal,
            "engagement_communication": engage_comm,
            "project_delivery": delivery,
            "leadership": leader,
            "culture": culture,
            "education": education,
            "summary": details.get("summary", ""),
            "met": "; ".join(details.get("met", [])),
            "missing": "; ".join(details.get("missing", [])),
        })

    # Save outputs
    cols = [
        "cv_id",
        "domain_finance",
        "technical_analytical",
        "engagement_communication",
        "project_delivery",
        "leadership",
        "culture",
        "education",
        "summary",
        "met",
        "missing",
    ]
    df = pd.DataFrame(rows, columns=cols)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    primary = safe_write_csv(df, OUT_CSV)
    run_copy = safe_write_csv(df, index_dir / f"ranked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    if callback:
        callback("Complete")

    return {"total": total, "processed": processed, "out_csv": primary, "run_csv": run_copy}

# ---------- CLI entry (optional) ----------
def main():
    res = run_ranking(callback=print)
    print(res)

if __name__ == "__main__":
    main()

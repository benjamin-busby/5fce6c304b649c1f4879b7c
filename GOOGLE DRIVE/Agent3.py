#!/usr/bin/env python3
"""
Agent3 — CV Cleaner
- Cleans .txt CVs: presentation tidy + email/phone deletion
- Safe for dates
- Reads from ./Raw Text, writes to ./Clean Text
- Outputs metadata.csv (and error_log.csv if needed)
"""

from pathlib import Path
from collections import Counter
import csv
import re
import unicodedata
from typing import Iterable, Optional, Dict, Any, Tuple, List

# --- PATHS RELATIVE TO SCRIPT ---
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "Raw Text"
OUTPUT_DIR = SCRIPT_DIR / "Clean Text"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- CLEANER FUNCTIONS ---
def clean_for_readability(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"[\u200B-\u200D\u2060\uFEFF]", "", t)
    t = "".join(ch for ch in t if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C")
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "•": "-", "∙": "-", "·": "-"
    }
    for a, b in replacements.items():
        t = t.replace(a, b)
    t = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", t)

    def norm_line(line: str) -> str:
        s = line.strip()
        s = re.sub(r"^([-\*•·∙]|[0-9]+\.)\s+", "- ", s)
        s = re.sub(r"\s+", " ", s)
        return s

    lines = [norm_line(ln) for ln in t.splitlines()]
    cleaned_lines = []
    blank = False
    for ln in lines:
        if ln == "":
            if not blank:
                cleaned_lines.append("")
                blank = True
        else:
            cleaned_lines.append(ln)
            blank = False
    return ("\n".join(cleaned_lines)).strip() + "\n"

def delete_contacts(text: str) -> Tuple[str, int, int]:
    email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    emails_found = email_re.findall(text)
    text = email_re.sub("", text)
    text = re.sub(r"\(\s*\+\d{1,3}\s*\)", "", text)
    text = re.sub(r"(?<!\w)\+\s*\d{1,3}(?!\d)", "", text)
    phone_cand = re.compile(r"(?<!\w)(?:\(\s*\d{2,5}\s*\)|\+?\s*\d)(?:[\d\s().-]{5,}\d)(?!\w)")

    def looks_like_date_range(s: str) -> bool:
        years = re.findall(r"\b(19|20)\d{2}\b", s)
        if len(years) >= 2: return True
        if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", s, re.I): return True
        if re.search(r"\b(present|current)\b", s, re.I): return True
        return False

    phones_found: List[str] = []

    def phone_replacer(m: re.Match) -> str:
        s = m.group(0)
        if looks_like_date_range(s): return s
        digits = re.sub(r"\D", "", s)
        if 9 <= len(digits) <= 15:
            phones_found.append(s)
            return ""
        return s

    text = phone_cand.sub(phone_replacer, text)
    return text, len(emails_found), len(phones_found)

# --- FILE I/O ---
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def process_file(p: Path) -> Dict[str, Any]:
    raw = read_text(p)
    pres = clean_for_readability(raw)
    stripped, n_emails, n_phones = delete_contacts(pres)
    final_text = clean_for_readability(stripped)
    final_text = re.sub(r"^[\s()\.\-]+$", "", final_text, flags=re.MULTILINE)
    final_text = clean_for_readability(final_text)
    out_path = OUTPUT_DIR / p.name
    write_text(out_path, final_text)
    return {
        "source_path": str(p),
        "output_path": str(out_path),
        "size_bytes": p.stat().st_size,
        "chars_raw": len(raw),
        "chars_clean": len(final_text),
        "emails_deleted": n_emails,
        "phones_deleted": n_phones,
    }

# --- CORE PIPELINE ---
def run_cleaning(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    exclude_files: Optional[Iterable[str]] = None,
    write_csv: bool = True,
) -> Dict[str, Any]:
    if not input_dir.exists():
        print(f"ERROR: INPUT FOLDER NOT FOUND: {input_dir.resolve()}")
        return {"processed":0,"skipped":0,"errors":1,"meta_rows":[],"err_rows":[],"err_csv":None}

    output_dir.mkdir(parents=True, exist_ok=True)
    excluded: set[str] = set(s.lower() for s in exclude_files) if exclude_files else set()
    files = [p for p in sorted(input_dir.rglob("*.txt")) if p.is_file()]

    processed, errors, skipped = 0, 0, 0
    error_types = Counter()
    meta_rows: List[Dict[str, Any]] = []
    err_rows: List[Dict[str, Any]] = []

    for p in files:
        fname, stem = p.name.lower(), p.stem.lower()
        if fname in excluded or stem in excluded or str(p).lower() in excluded:
            skipped += 1
            continue
        try:
            row = process_file(p)
            meta_rows.append(row)
            processed += 1
        except Exception as e:
            errors += 1
            et = type(e).__name__
            error_types[et] += 1
            err_rows.append({"source_path": str(p),"error_type": et,"error_msg": str(e)})

    meta_csv = output_dir / "metadata.csv"
    if write_csv:
        with meta_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "source_path","output_path","size_bytes",
                "chars_raw","chars_clean","emails_deleted","phones_deleted"
            ])
            w.writeheader(); w.writerows(meta_rows)

    err_csv = None
    if err_rows and write_csv:
        err_csv = output_dir / "error_log.csv"
        with err_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source_path","error_type","error_msg"])
            w.writeheader(); w.writerows(err_rows)

    print(f"PROCESSED: {processed} | SKIPPED: {skipped} | ERRORS: {errors}")
    return {
        "processed": processed, "skipped": skipped, "errors": errors,
        "meta_rows": meta_rows, "err_rows": err_rows,
        "meta_csv": str(meta_csv), "err_csv": str(err_csv) if err_rows else None,
        "error_summary_by_type": dict(error_types),
    }

# --- AUTO-EXCLUDE AGENT1 BAD FILES ---
def run_cleaning_excluding_agent1(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    write_csv: bool = True,
) -> Dict[str, Any]:
    try:
        import Agent1
        bad_files = Agent1.run_and_get_bad_files(callback=None)
    except Exception as e:
        print(f"Failed to import Agent1 or get bad files: {e}")
        bad_files = None
    return run_cleaning(input_dir=input_dir, output_dir=output_dir, exclude_files=bad_files, write_csv=write_csv)

# --- CLI ENTRY ---
def main():
    run_cleaning(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, exclude_files=None, write_csv=True)

if __name__ == "__main__":
    main()

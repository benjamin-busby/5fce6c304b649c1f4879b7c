#!/usr/bin/env python3
# CLEAN TXT CVS: PRESENTATION TIDY + DELETE EMAILS & PHONE NUMBERS (SAFE FOR DATES)
# READS .TXT FILES FROM ./Raw Text
# WRITES CLEANED FILES TO ./Clean Text
# EMITS METADATA.CSV (AND ERROR_LOG.CSV IF NEEDED)

from pathlib import Path                 # FILESYSTEM PATH HANDLING
from collections import Counter          # SIMPLE COUNTER FOR ERROR TYPES
import csv                               # CSV OUTPUT FOR METADATA/ERRORS
import re                                # REGEX FOR TEXT CLEANING/REDACTION
import unicodedata                       # UNICODE NORMALISATION
from typing import Iterable, Optional, Dict, Any, Tuple, List

# --- PATHS (ANCHORED TO THIS SCRIPT) ---
SCRIPT_DIR = Path(__file__).resolve().parent   # ABSOLUTE DIR OF THIS SCRIPT
INPUT_DIR  = SCRIPT_DIR / "Raw Text"           # SOURCE .TXT FILES   (fixed: no leading slash)
OUTPUT_DIR = SCRIPT_DIR / "Clean Text"         # DESTINATION CLEANED .TXT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # ENSURE OUTPUT FOLDER EXISTS

# --- PRESENTATION-ONLY CLEANER ---
def clean_for_readability(text: str) -> str:
    # UNICODE NORMALISATION TO NFKC (COMPATIBILITY FORM)
    t = unicodedata.normalize("NFKC", text)
    # REMOVE ZERO-WIDTH CHARACTERS (COMMON EXTRACTOR ARTIFACTS)
    t = re.sub(r"[\u200B-\u200D\u2060\uFEFF]", "", t)
    # STRIP OTHER CONTROL CHARS BUT KEEP NEWLINES/TABS FOR STRUCTURE
    t = "".join(ch for ch in t if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C")

    # STANDARDISE “SMART” PUNCTUATION AND BULLET GLYPHS TO ASCII
    replacements = {
        "\u2018": "'",  # LEFT SINGLE QUOTE → APOSTROPHE
        "\u2019": "'",  # RIGHT SINGLE QUOTE → APOSTROPHE
        "\u201C": '"',  # LEFT DOUBLE QUOTE → QUOTE
        "\u201D": '"',  # RIGHT DOUBLE QUOTE → QUOTE
        "\u2013": "-",  # EN DASH → HYPHEN
        "\u2014": "-",  # EM DASH → HYPHEN
        "\u2212": "-",  # MINUS SIGN → HYPHEN
        "•": "-",       # BULLET → HYPHEN
        "∙": "-",       # BULLET VARIANT → HYPHEN
        "·": "-",       # MIDDLE DOT → HYPHEN
    }
    for a, b in replacements.items():
        t = t.replace(a, b)

    # FIX HYPHENATION BROKEN ACROSS LINE BREAKS: "WORD-\nWORD" → "WORDWORD"
    t = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", t)

    # NORMALISE BULLET PREFIXES AND COMPRESS INTERNAL SPACES PER LINE
    def norm_line(line: str) -> str:
        s = line.strip()
        s = re.sub(r"^([-\*•·∙]|[0-9]+\.)\s+", "- ", s)  # ANY BULLET/NUMBER → "- "
        s = re.sub(r"\s+", " ", s)                      # MULTIPLE SPACES → SINGLE
        return s

    lines = [norm_line(ln) for ln in t.splitlines()]

    # COLLAPSE MULTIPLE BLANK LINES (KEEP AT MOST ONE)
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

    # JOIN LINES, STRIP OUTER WHITESPACE, ENSURE SINGLE TRAILING NEWLINE
    return ("\n".join(cleaned_lines)).strip() + "\n"

# --- DELETE EMAILS AND PHONE NUMBERS (WITH DATE-SAFE HEURISTICS) ---
def delete_contacts(text: str) -> Tuple[str, int, int]:
    # 1) DELETE EMAILS (STANDARD USER@DOMAIN.TLD)
    email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    emails_found = email_re.findall(text)
    text = email_re.sub("", text)

    # 2) OPTIONAL FIRST PASS: REMOVE (+NN) AND +NN COUNTRY CODES ONLY
    text = re.sub(r"\(\s*\+\d{1,3}\s*\)", "", text)                 # ( +27 ) OR (+44)
    text = re.sub(r"(?<!\w)\+\s*\d{1,3}(?!\d)", "", text)           # bare +27/+44

    # 3) PHONE CANDIDATES
    phone_cand = re.compile(r"(?<!\w)(?:\(\s*\d{2,5}\s*\)|\+?\s*\d)(?:[\d\s().-]{5,}\d)(?!\w)")

    def looks_like_date_range(s: str) -> bool:
        years = re.findall(r"\b(19|20)\d{2}\b", s)
        if len(years) >= 2:
            return True
        if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", s, re.I):
            return True
        if re.search(r"\b(present|current)\b", s, re.I):
            return True
        return False

    phones_found: List[str] = []

    def phone_replacer(m: re.Match) -> str:
        s = m.group(0)
        if looks_like_date_range(s):
            return s
        digits = re.sub(r"\D", "", s)
        if 9 <= len(digits) <= 15:
            phones_found.append(s)
            return ""
        return s

    text = phone_cand.sub(phone_replacer, text)
    return text, len(emails_found), len(phones_found)

# --- SIMPLE FILE IO HELPERS ---
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# --- PROCESS A SINGLE FILE ---
def process_file(p: Path) -> Dict[str, Any]:
    raw = read_text(p)                               # LOAD RAW TXT
    pres = clean_for_readability(raw)                # APPLY PRESENTATION CLEANING
    stripped, n_emails, n_phones = delete_contacts(pres)  # DELETE EMAILS/PHONES (SAFE)
    final_text = clean_for_readability(stripped)     # FINAL TIDY AFTER DELETIONS

    # DROP LINES THAT BECAME JUST PUNCTUATION/BRACKETS AFTER DELETION
    final_text = re.sub(r"^[\s()\.\-]+$", "", final_text, flags=re.MULTILINE)
    final_text = clean_for_readability(final_text)   # ONE MORE PASS TO COLLAPSE GAPS

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

# --- CORE PIPELINE (IMPORT-FRIENDLY) ---
def run_cleaning(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    exclude_files: Optional[Iterable[str]] = None,
    write_csv: bool = True,
) -> Dict[str, Any]:
    """
    Cleans all .txt files in input_dir into output_dir, optionally skipping files
    whose filename or stem appears in exclude_files.

    Returns a dict with processed, skipped, errors, metadata paths, and rows.
    """
    if not input_dir.exists():
        print(f"ERROR: INPUT FOLDER NOT FOUND: {input_dir.resolve()}")
        return {
            "processed": 0, "skipped": 0, "errors": 1,
            "meta_rows": [], "err_rows": [],
            "err_csv": None,
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalise exclusions
    excluded: set[str] = set()
    if exclude_files:
        for s in exclude_files:
            excluded.add(str(s).lower())

    # Enumerate input .txt
    files = [p for p in sorted(input_dir.rglob("*.txt")) if p.is_file()]

    ##print(f"INPUT DIR:  {input_dir.resolve()}")
    ##print(f"OUTPUT DIR: {output_dir.resolve()}")
    ##print(f"FOUND {len(files)} .TXT FILES.")

    processed, errors, skipped = 0, 0, 0
    error_types = Counter()
    meta_rows: List[Dict[str, Any]] = []
    err_rows: List[Dict[str, Any]] = []

    for p in files:
        fname = p.name.lower()
        stem  = p.stem.lower()
        if fname in excluded or stem in excluded or str(p).lower() in excluded:
            skipped += 1
            print(f"– SKIP {p.name} (in exclude list)")
            continue

        try:
            row = process_file(p)
            meta_rows.append(row)
            processed += 1
            ##print(
            ##    f"✓ {p.name}  -> CLEANED "
            ##    f"({row['chars_raw']}→{row['chars_clean']} CHARS, "
            ##    f"EMAILS DELETED: {row['emails_deleted']}, "
            ##    f"PHONES DELETED: {row['phones_deleted']})"
            ##)
        except Exception as e:
            errors += 1
            et = type(e).__name__
            error_types[et] += 1
            err_rows.append({
                "source_path": str(p),
                "error_type": et,
                "error_msg": str(e),
            })
            ##print(f"✗ {p.name}: {e}")

    # WRITE METADATA CSV
    meta_csv = output_dir / "metadata.csv"
    if write_csv:
        with meta_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "source_path", "output_path", "size_bytes",
                    "chars_raw", "chars_clean", "emails_deleted", "phones_deleted"
                ],
            )
            w.writeheader(); w.writerows(meta_rows)

    # WRITE ERROR LOG CSV IF ANY
    err_csv = None
    if err_rows and write_csv:
        err_csv = output_dir / "error_log.csv"
        with err_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source_path", "error_type", "error_msg"])
            w.writeheader(); w.writerows(err_rows)

    # SUMMARY
    print("\n--- SUMMARY ---")
    print(f"PROCESSED: {processed}  |  SKIPPED: {skipped}  |  ERRORS: {errors}")

    if err_rows:
        ##print("\n--- ERROR SUMMARY (BY TYPE) ---")
        for et, cnt in error_types.most_common():
            pass
            ##print(f"{et}: {cnt}")

    return {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "meta_rows": meta_rows,
        "err_rows": err_rows,
        "meta_csv": str(meta_csv),
        "err_csv": str(err_csv) if err_rows else None,
        "error_summary_by_type": dict(error_types),
    }

# --- Convenience: auto-exclude using Agent1's bad_files ---
def run_cleaning_excluding_agent1(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    write_csv: bool = True,
) -> Dict[str, Any]:
    """
    Imports Agent1 to obtain its bad_files list, then runs cleaning with those excluded.
    """
    try:
        import Agent1
    except Exception as e:
        print(f"Could not import Agent1 to obtain exclusions: {e}")
        return run_cleaning(input_dir=input_dir, output_dir=output_dir, exclude_files=None, write_csv=write_csv)

    try:
        bad_files = Agent1.run_and_get_bad_files(callback=None)
    except Exception as e:
        print(f"Failed to obtain bad files from Agent1: {e}")
        bad_files = None

    return run_cleaning(input_dir=input_dir, output_dir=output_dir, exclude_files=bad_files, write_csv=write_csv)

# --- CLI ENTRY POINT ---
def main():
    # Default: run without exclusions.
    run_cleaning(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, exclude_files=None, write_csv=True)

if __name__ == "__main__":
    main()

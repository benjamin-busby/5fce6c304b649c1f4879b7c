# Agent2.py
# Text extraction for each CV, import-friendly API, and ability to exclude files
# (e.g., bad files identified by Agent1).

from pathlib import Path
import csv
from collections import Counter
from typing import Iterable, Optional, Dict, Any, Tuple, List

from pypdf import PdfReader
import docx2txt
import chardet

# ---- Defaults (override when calling) ----
DEFAULT_CV_DIRECTORY = Path(
    r"C:\Users\BenjaminBusby\OneDrive - MONOCLE SOLUTIONS (PTY) LTD\Documents\Monocle AI Challenge\CV Database"
)
DEFAULT_OUTPUT_DIRECTORY = Path("Raw Text")  # created under current working directory
SUPPORTED_EXTS = {".pdf", ".docx", ".txt"}

# ---- Readers ----
def read_pdf(path: Path) -> Tuple[str, int]:
    text_parts: List[str] = []
    with path.open("rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        pages = len(reader.pages)
    return "\n".join(text_parts), pages

def read_docx(path: Path) -> str:
    return docx2txt.process(str(path)) or ""

def read_txt(path: Path) -> str:
    raw = path.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    return raw.decode(enc, errors="ignore")

# ---- Core processing function ----
def process_cvs(
    cv_directory: Path = DEFAULT_CV_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    exclude_sources: Optional[Iterable[str]] = None,
    write_csv: bool = True,
) -> Dict[str, Any]:
    """
    Process CVs from cv_directory and write extracted text files to output_directory.

    Parameters
    ----------
    cv_directory : Path
        Root folder to search for CV files (recursively).
    output_directory : Path
        Folder where .txt outputs and metadata/error CSVs will be written.
    exclude_sources : Iterable[str] or None
        Filenames (e.g. "CV_0457.pdf") or stems (e.g. "CV_0457") to skip.
    write_csv : bool
        If True, writes metadata.csv and error_log.csv (if any).

    Returns
    -------
    dict with keys: processed, errors, skipped, meta_rows, error_rows, meta_csv, err_csv, error_summary_by_type
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    excluded: set[str] = set()
    if exclude_sources:
        for s in exclude_sources:
            excluded.add(str(s).lower())

    files = [p for p in cv_directory.rglob("*") if p.is_file()]

    meta_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    error_types: Counter[str] = Counter()

    processed = 0
    errors = 0
    skipped = 0

    for p in sorted(files):
        ext = p.suffix.lower()

        # Exclusion by full path or stem or filename
        if str(p).lower() in excluded or p.stem.lower() in excluded or p.name.lower() in excluded:
            skipped += 1
            continue

        # Skip unsupported types silently
        if ext not in SUPPORTED_EXTS:
            skipped += 1
            continue

        try:
            if ext == ".pdf":
                text, pages = read_pdf(p)
            elif ext == ".docx":
                text, pages = read_docx(p), None
            else:  # ".txt"
                text, pages = read_txt(p), None

            out_path = output_directory / (p.stem + ".txt")
            out_path.write_text(text, encoding="utf-8")

            meta_rows.append({
                "source_path": str(p),
                "output_path": str(out_path),
                "ext": ext,
                "size_bytes": p.stat().st_size,
                "chars": len(text),
                "pdf_pages": pages if pages is not None else "",
            })
            processed += 1

        except Exception as e:
            errors += 1
            etype = type(e).__name__
            error_types[etype] += 1
            error_rows.append({
                "source_path": str(p),
                "ext": ext,
                "size_bytes": p.stat().st_size if p.exists() else "",
                "error_type": etype,
                "error_msg": str(e),
            })

    meta_csv = output_directory / "metadata.csv"
    err_csv = output_directory / "error_log.csv"

    if write_csv:
        with meta_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["source_path", "output_path", "ext", "size_bytes", "chars", "pdf_pages"]
            )
            w.writeheader()
            w.writerows(meta_rows)

        if error_rows:
            with err_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["source_path", "ext", "size_bytes", "error_type", "error_msg"]
                )
                w.writeheader()
                w.writerows(error_rows)

    return {
        "processed": processed,
        "errors": errors,
        "skipped": skipped,
        "meta_rows": meta_rows,
        "error_rows": error_rows,
        "meta_csv": str(meta_csv),
        "err_csv": str(err_csv) if error_rows else None,
        "error_summary_by_type": dict(error_types),
    }

# ---- GUI-friendly convenience summary ----
def run_text_extraction(
    cv_directory: Path = DEFAULT_CV_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    exclude_files: Optional[Iterable[str]] = None,
    write_csv: bool = True,
) -> str:
    res = process_cvs(
        cv_directory=cv_directory,
        output_directory=output_directory,
        exclude_sources=exclude_files,
        write_csv=write_csv,
    )
    summary = f"Processed: {res['processed']}  |  Skipped: {res['skipped']}  |  Errors: {res['errors']}"
    if res["error_rows"]:
        et = res["error_summary_by_type"]
        if et:
            details = ", ".join(f"{k}:{v}" for k, v in et.items())
            summary += f"\nError Summary (by type): {details}"
    return summary

# ---- Optional: pull bad files from Agent1 and exclude automatically ----
def run_text_extraction_excluding_agent1(
    cv_directory: Path = DEFAULT_CV_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    write_csv: bool = True,
) -> str:
    try:
        import Agent1
    except Exception as e:
        return f"Could not import Agent1 to obtain exclusions: {e}"
    try:
        bad_files = Agent1.run_and_get_bad_files(callback=None)
    except Exception as e:
        return f"Failed to obtain bad files from Agent1: {e}"
    return run_text_extraction(
        cv_directory=cv_directory,
        output_directory=output_directory,
        exclude_files=bad_files,
        write_csv=write_csv,
    )

if __name__ == "__main__":
    print(run_text_extraction())

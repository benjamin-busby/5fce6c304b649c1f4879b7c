# streamlit_app.py
# Streamlit UI for your CV Analysis pipeline (Agents 1–5)
from __future__ import annotations
import os, sys
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title="CV Analysis", layout="wide")
st.title("CV Analysis – RAG Pipeline")

# ---------- Locate agents & data folders ----------
BASE = Path(__file__).resolve().parent

def find_agents_dir() -> Path:
    # Prefer ./app/, else repo root, else GOOGLE DRIVE/
    candidates = [BASE / "app", BASE, BASE / "GOOGLE DRIVE"]
    for d in candidates:
        if (d / "Agent1.py").exists():
            return d
    # last resort: the one that has Agent4/5
    for d in candidates:
        for n in ["Agent4.py", "Agent5.py", "Agent2.py", "Agent3.py"]:
            if (d / n).exists():
                return d
    return BASE

AGENTS_DIR = find_agents_dir()
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

RAW_DIR   = AGENTS_DIR / "Raw Text"
CLEAN_DIR = AGENTS_DIR / "Clean Text"
INDEX_DIR = AGENTS_DIR / "Index"

def find_cv_db_dir() -> Path:
    # case-insensitive fallback for your "CV Database"
    candidates = [
        BASE / "CV Database",
        BASE / "CV database",
        AGENTS_DIR.parent / "CV Database",
        AGENTS_DIR.parent / "CV database",
    ]
    for p in candidates:
        if p.exists():
            return p
    return BASE / "CV Database"

# ---------- Sidebar Controls ----------
st.sidebar.header("Folders")
cv_dir = st.sidebar.text_input("CV folder", value=str(find_cv_db_dir()))
cv_path = Path(cv_dir)

with st.sidebar.expander("Advanced"):
    st.caption("Agents directory (auto-detected):")
    st.code(str(AGENTS_DIR))
    st.caption("Raw/Clean/Index dirs (auto):")
    st.code(f"RAW_DIR   = {RAW_DIR}\nCLEAN_DIR = {CLEAN_DIR}\nINDEX_DIR = {INDEX_DIR}")

st.sidebar.header("Actions")
run_gatekeep = st.sidebar.button("1) Gatekeep (Agent1)")
run_extract  = st.sidebar.button("2) Extract Text (Agent2)")
run_clean    = st.sidebar.button("3) Clean Text (Agent3)")
run_vector   = st.sidebar.button("4) Vectorise (Agent4)")
run_rank     = st.sidebar.button("5) Rank (Agent5)")
show_report  = st.sidebar.button("6) Analyse / Report")

# keep bad files between steps
if "agent1_bad_files" not in st.session_state:
    st.session_state.agent1_bad_files = []

# ---------- Helpers ----------
def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

def inject_azure_secrets():
    # Read Azure OpenAI creds from Streamlit secrets if present
    for k in ["AZURE_OPENAI_API_KEY",
              "AZURE_OPENAI_ENDPOINT",
              "AZURE_OPENAI_DEPLOYMENT",
              "AZURE_OPENAI_API_VERSION"]:
        if k in st.secrets:
            os.environ[k] = str(st.secrets[k])

# ---------- Main panel ----------
st.write("### Pipeline Overview")
st.markdown(
    """
    1. **Gatekeep** scans CVs and filters out non-English/too short/not-a-CV  
    2. **Extract Text** converts PDFs/DOCX/TXT → raw TXT  
    3. **Clean Text** tidies text and redacts emails/phones  
    4. **Vectorise** builds embeddings & FAISS index from cleaned text  
    5. **Rank (LLM)** scores CVs with Azure OpenAI using retrieved snippets  
    6. **Analyse / Report** shows ranked results and summary visuals  
    """
)

ensure_dirs()

# ---------- 1) Agent1: Gatekeep ----------
if run_gatekeep:
    with st.status("Running Agent1 (Gatekeeper)…", expanded=True) as status:
        try:
            import Agent1
            # Fix the CV folder path (your code used 'CV database' — watch case)
            Agent1.CV_DIRECTORY = Path(cv_path)
            logs = []

            def cb(msg: str):
                logs.append(msg)

            res = Agent1.gatekeep(callback=cb)
            st.code("\n".join(logs) or "No logs.")
            st.session_state.agent1_bad_files = res.get("bad_files", [])
            status.update(label="Agent1 complete", state="complete")
            st.success(f"Processed: {res['processed']} | Rejected: {res['rejected']}")
            if res.get("error_types"):
                st.info(f"Error types: {res['error_types']}")
        except Exception as e:
            status.update(label="Agent1 failed", state="error")
            st.exception(e)

# ---------- 2) Agent2: Extract ----------
if run_extract:
    with st.status("Running Agent2 (Text Extraction)…", expanded=True) as status:
        try:
            import Agent2
            summary = Agent2.run_text_extraction(
                cv_directory=Path(cv_path),
                output_directory=RAW_DIR,
                exclude_files=st.session_state.agent1_bad_files,
                write_csv=True,
            )
            status.update(label="Agent2 complete", state="complete")
            st.success(summary)
            meta = RAW_DIR / "metadata.csv"
            if meta.exists():
                st.dataframe(pd.read_csv(meta).head(50))
        except Exception as e:
            status.update(label="Agent2 failed", state="error")
            st.exception(e)

# ---------- 3) Agent3: Clean ----------
if run_clean:
    with st.status("Running Agent3 (Cleaner)…", expanded=True) as status:
        try:
            import Agent3
            res = Agent3.run_cleaning(
                input_dir=RAW_DIR,
                output_dir=CLEAN_DIR,
                exclude_files=st.session_state.agent1_bad_files,
                write_csv=True,
            )
            status.update(label="Agent3 complete", state="complete")
            st.success(f"Processed: {res['processed']} | Errors: {res['errors']}")
            meta = CLEAN_DIR / "metadata.csv"
            if meta.exists():
                st.dataframe(pd.read_csv(meta).head(50))
        except Exception as e:
            status.update(label="Agent3 failed", state="error")
            st.exception(e)

# ---------- 4) Agent4: Vectorise (FAISS) ----------
if run_vector:
    with st.status("Running Agent4 (Vectorise/Index)…", expanded=True) as status:
        try:
            import Agent4
            out = Agent4.build_index(
                clean_dir=CLEAN_DIR,
                index_dir=INDEX_DIR,
                verbose=True,
                callback=lambda m: st.write(m),
            )
            status.update(label="Agent4 complete", state="complete")
            st.success(f"Files: {out['files']} | Chunks: {out['chunks']}")
            st.write(f"Index: {out['index_path']}")
            meta = INDEX_DIR / "metadata.csv"
            if meta.exists():
                st.dataframe(pd.read_csv(meta).head(50))
        except Exception as e:
            status.update(label="Agent4 failed", state="error")
            st.exception(e)

# ---------- 5) Agent5: Rank (Azure OpenAI) ----------
if run_rank:
    with st.status("Running Agent5 (LLM Ranking)…", expanded=True) as status:
        try:
            inject_azure_secrets()
            import Agent5
            res = Agent5.run_ranking(
                index_dir=INDEX_DIR,
                callback=lambda m: st.write(m),
            )
            status.update(label="Agent5 complete", state="complete")
            st.success(res)
            # offer download
            out_csv = Path(res.get("out_csv" or "")) if isinstance(res, dict) else None
            if out_csv and out_csv.exists():
                st.download_button(
                    "Download ranked.csv",
                    data=out_csv.read_bytes(),
                    file_name="ranked.csv",
                )
        except Exception as e:
            status.update(label="Agent5 failed", state="error")
            st.exception(e)

# ---------- 6) Analyse / Report ----------
if show_report:
    st.subheader("Ranked Results")
    ranked_csv = INDEX_DIR / "ranked.csv"
    if not ranked_csv.exists():
        st.warning("No ranked.csv found yet. Run step 5 first.")
    else:
        df = pd.read_csv(ranked_csv)
        st.dataframe(df)
        # Simple summary visuals
        score_cols = [
            c for c in df.columns
            if c in ["domain_finance","technical_analytical","engagement_communication",
                     "project_delivery","leadership","culture","education"]
        ]
        if score_cols:
            df["average_score"] = df[score_cols].mean(axis=1).round(1)
            st.write("Top by average score:")
            top = df.sort_values("average_score", ascending=False).head(10)
            st.bar_chart(top.set_index("cv_id")["average_score"])
        else:
            st.info("No score columns detected.")

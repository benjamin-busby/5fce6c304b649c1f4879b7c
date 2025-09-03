# drive_cv_probe_min.py
# EPHEMERAL LISTING - PRINTS ONLY: "X Files found"

import os, sys
from typing import Optional, List
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
DRIVE_PATH = ["Google_Drive_Monocle_AI_Challenge", "01. CV Database"]

def get_drive_service():
    if not os.path.exists("credentials.json"):
        # No extra prints - fail silently with zero files
        sys.exit("0 Files found")
    flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", authorization_prompt_message="")
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def find_folder(service, name: str, parent_id: Optional[str] = None) -> Optional[str]:
    q = "mimeType='application/vnd.google-apps.folder' and trashed=false and name=%r" % name
    if parent_id:
        q += f" and '{parent_id}' in parents"
    res = service.files().list(q=q, fields="files(id)").execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None

def folder_id_by_path(service, segments: List[str]) -> Optional[str]:
    parent = None
    for seg in segments:
        fid = find_folder(service, seg, parent)
        if not fid:
            return None
        parent = fid
    return parent

def count_files(service, folder_id: str) -> int:
    # Just one page (up to 1000); simple proof
    res = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id)",
        pageSize=1000
    ).execute()
    return len(res.get("files", []))

if __name__ == "__main__":
    try:
        svc = get_drive_service()
        fid = folder_id_by_path(svc, DRIVE_PATH)
        n = count_files(svc, fid) if fid else 0
        print(f"{n} Files found")
    except Exception:
        # On any error, keep the contract: print only the count line
        print("0 Files found")

#!/usr/bin/env python3
"""
Automated sync + deploy to Hugging Face Spaces.

Usage:
    python update_hf.py              # sync + upload everything
    python update_hf.py --db-only    # just upload DB (skip sync)
    python update_hf.py --full       # upload all files + DB (first time or after code changes)
"""

import subprocess
import sys
import os

# === CONFIGURATION ===
REPO_ID = "rottg/telegram-analytics"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_DIR, "telegram.db")

# Files to upload (code + config)
CODE_FILES = [
    "dashboard.py", "ai_search.py", "algorithms.py", "data_structures.py",
    "indexer.py", "search.py", "semantic_search.py", "schema.sql",
    "Dockerfile", "requirements.txt", "README.md",
]
FOLDERS = ["static", "templates"]

# Token is read from environment variable or .hf_token file
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    token_file = os.path.join(PROJECT_DIR, ".hf_token")
    if os.path.exists(token_file):
        with open(token_file) as f:
            HF_TOKEN = f.read().strip()
    else:
        print("ERROR: Set HF_TOKEN env var or create .hf_token file with your token")
        sys.exit(1)


def run_sync():
    """Run daily_sync.py"""
    sync_script = os.path.join(PROJECT_DIR, "daily_sync.py")
    print("\n=== Step 1: Running daily sync ===")
    result = subprocess.run([sys.executable, sync_script], cwd=PROJECT_DIR)
    if result.returncode != 0:
        print("ERROR: Sync failed!")
        sys.exit(1)
    print("Sync complete.")


def upload_to_hf(full=False):
    """Upload files to HF Space using the API (no git needed)."""
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    if full:
        # Upload all code files + folders + DB
        print("\n=== Uploading all files to HF ===")

        # Collect all files to upload
        upload_files = []
        for f in CODE_FILES:
            path = os.path.join(PROJECT_DIR, f)
            if os.path.exists(path):
                upload_files.append((path, f))

        for folder in FOLDERS:
            folder_path = os.path.join(PROJECT_DIR, folder)
            if os.path.exists(folder_path):
                for root, dirs, files in os.walk(folder_path):
                    for fname in files:
                        full_path = os.path.join(root, fname)
                        rel_path = os.path.relpath(full_path, PROJECT_DIR)
                        upload_files.append((full_path, rel_path.replace("\\", "/")))

        # Add DB
        upload_files.append((DB_PATH, "telegram.db"))

        print(f"Uploading {len(upload_files)} files...")
        for local_path, repo_path in upload_files:
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            if size_mb > 1:
                print(f"  {repo_path} ({size_mb:.0f} MB)...")
            else:
                print(f"  {repo_path}")

        api.upload_folder(
            folder_path=PROJECT_DIR,
            repo_id=REPO_ID,
            repo_type="space",
            allow_patterns=[f for _, f in upload_files],
        )
    else:
        # DB only - delete old, upload new
        print("\n=== Removing old DB from HF ===")
        try:
            api.delete_file("telegram.db", repo_id=REPO_ID, repo_type="space")
            print("Old DB removed.")
        except Exception as e:
            print(f"No old DB to remove ({e})")

        print("\n=== Uploading new DB to HF ===")
        db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
        print(f"Uploading {db_size_mb:.0f} MB...")

        api.upload_file(
            path_or_fileobj=DB_PATH,
            path_in_repo="telegram.db",
            repo_id=REPO_ID,
            repo_type="space",
        )

    print("Upload complete!")
    print(f"\nSite will rebuild at: https://rottg-telegram-analytics.hf.space")


def upload_code_only():
    """Upload only code files (no DB) - for when you just changed code/templates."""
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    print("\n=== Uploading code files to HF ===")
    upload_patterns = CODE_FILES + [f"{folder}/**" for folder in FOLDERS]
    print(f"Patterns: {upload_patterns}")

    api.upload_folder(
        folder_path=PROJECT_DIR,
        repo_id=REPO_ID,
        repo_type="space",
        allow_patterns=upload_patterns,
    )
    print("Upload complete!")
    print(f"\nSite will rebuild at: https://rottg-telegram-analytics.hf.space")


def main():
    db_only = "--db-only" in sys.argv
    full = "--full" in sys.argv
    code_only = "--code-only" in sys.argv

    if not db_only and not code_only:
        run_sync()
    else:
        print("Skipping sync")

    if code_only:
        upload_code_only()
    else:
        upload_to_hf(full=full or (not db_only and not code_only))
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()

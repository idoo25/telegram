#!/usr/bin/env python3
"""
Automated sync + deploy to Hugging Face.

DB goes to Dataset repo (rottg/telegram-db) - no LFS limits issue.
Code goes to Space repo (rottg/telegram-analytics).
After DB upload, Space is restarted to load the new data.

Usage:
    python update_hf.py              # sync + upload DB + restart Space
    python update_hf.py --db-only    # just upload DB + restart (skip sync)
    python update_hf.py --code-only  # just upload code (no restart needed, triggers rebuild)
    python update_hf.py --full       # sync + upload DB + code + restart
    python update_hf.py --no-restart # skip the Space restart
"""

import subprocess
import sys
import os

# === CONFIGURATION ===
SPACE_REPO = "rottg/telegram-analytics"
DATASET_REPO = "rottg/telegram-db"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_DIR, "telegram.db")

# Files to upload to Space (code + config, NO DB)
CODE_FILES = [
    "dashboard.py", "ai_search.py", "algorithms.py", "data_structures.py",
    "indexer.py", "search.py", "semantic_search.py", "hybrid_search.py",
    "gemini_client.py", "stylometry.py", "schema.sql", "Dockerfile", "requirements.txt", "README.md",
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


def upload_db():
    """Upload DB to the Dataset repo (separate from Space)."""
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    print("\n=== Uploading DB to Dataset repo ===")
    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"Uploading {db_size_mb:.0f} MB to {DATASET_REPO}...")

    api.upload_file(
        path_or_fileobj=DB_PATH,
        path_in_repo="telegram.db",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="Update telegram.db"
    )

    print("✓ DB uploaded to Dataset repo!")
    print(f"  https://huggingface.co/datasets/{DATASET_REPO}")


def upload_code():
    """Upload code files to the Space repo (no DB)."""
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    print("\n=== Uploading code to Space repo ===")
    upload_patterns = CODE_FILES + [f"{folder}/**" for folder in FOLDERS]
    print(f"Patterns: {upload_patterns}")

    api.upload_folder(
        folder_path=PROJECT_DIR,
        repo_id=SPACE_REPO,
        repo_type="space",
        allow_patterns=upload_patterns,
        commit_message="Update code"
    )

    print("✓ Code uploaded to Space!")
    print(f"  Site will rebuild at: https://rottg-telegram-analytics.hf.space")


def restart_space():
    """Restart the Space to load the new DB."""
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    print("\n=== Restarting Space to load new DB ===")
    api.restart_space(repo_id=SPACE_REPO)
    print("✓ Space restart triggered!")
    print(f"  Site will be back in ~30 seconds: https://rottg-telegram-analytics.hf.space")


def main():
    db_only = "--db-only" in sys.argv
    code_only = "--code-only" in sys.argv
    full = "--full" in sys.argv
    no_restart = "--no-restart" in sys.argv

    print("=" * 50)
    print("HuggingFace Update Script")
    print("=" * 50)

    # Run sync unless skipped
    if not db_only and not code_only:
        run_sync()
    else:
        print("\nSkipping sync")

    # Upload DB to Dataset repo
    if not code_only:
        upload_db()

    # Upload code to Space repo
    if code_only or full:
        upload_code()

    # Restart Space to load new DB (unless --no-restart or --code-only)
    if not code_only and not no_restart:
        restart_space()

    print("\n" + "=" * 50)
    print("✓ Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()

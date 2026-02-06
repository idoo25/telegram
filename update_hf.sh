#!/bin/bash
# Update Hugging Face Spaces with latest database
#
# Usage:
#   ./update_hf.sh                    # sync + update HF
#   ./update_hf.sh --db-only          # just update HF (skip sync)
#
# Prerequisites:
#   1. Clone your HF Space repo next to this project:
#      cd ~ && git clone https://huggingface.co/spaces/USERNAME/telegram-analytics
#   2. Install Git LFS: git lfs install
#
# Adjust these paths if needed:
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
HF_DIR="$HOME/telegram-analytics"

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Telegram Analytics - HF Spaces Updater ===${NC}"

# Check HF repo exists
if [ ! -d "$HF_DIR/.git" ]; then
    echo -e "${RED}Error: HF Space repo not found at $HF_DIR${NC}"
    echo "Clone it first:"
    echo "  git clone https://huggingface.co/spaces/USERNAME/telegram-analytics $HF_DIR"
    exit 1
fi

# Step 1: Run daily sync (unless --db-only)
if [ "$1" != "--db-only" ]; then
    echo -e "\n${YELLOW}Step 1: Running daily sync...${NC}"
    cd "$PROJECT_DIR"
    python daily_sync.py
    echo -e "${GREEN}Sync complete.${NC}"
else
    echo -e "\n${YELLOW}Step 1: Skipping sync (--db-only)${NC}"
fi

# Step 2: Copy DB to HF repo
echo -e "\n${YELLOW}Step 2: Copying telegram.db to HF repo...${NC}"
cp "$PROJECT_DIR/telegram.db" "$HF_DIR/telegram.db"
echo -e "${GREEN}Copied $(du -h "$HF_DIR/telegram.db" | cut -f1) to HF repo.${NC}"

# Step 3: Copy code files (in case they changed)
echo -e "\n${YELLOW}Step 3: Copying code files...${NC}"
for f in dashboard.py ai_search.py algorithms.py data_structures.py indexer.py search.py semantic_search.py schema.sql Dockerfile requirements.txt README.md; do
    if [ -f "$PROJECT_DIR/$f" ]; then
        cp "$PROJECT_DIR/$f" "$HF_DIR/$f"
    fi
done
cp -r "$PROJECT_DIR/static/" "$HF_DIR/static/"
cp -r "$PROJECT_DIR/templates/" "$HF_DIR/templates/"
echo -e "${GREEN}Code files copied.${NC}"

# Step 4: Commit and push
echo -e "\n${YELLOW}Step 4: Pushing to Hugging Face...${NC}"
cd "$HF_DIR"
git add .
if git diff --cached --quiet; then
    echo -e "${YELLOW}No changes to push.${NC}"
else
    git commit -m "Update DB $(date +%Y-%m-%d)"
    git push
    echo -e "${GREEN}Pushed to HF Spaces. Rebuilding will start automatically.${NC}"
fi

echo -e "\n${GREEN}=== Done! ===${NC}"

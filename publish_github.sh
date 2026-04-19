#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/regina/Desktop/5360 project"
GH_BIN="$PROJECT_DIR/.tools/gh_2.90.0_macOS_arm64/bin/gh"
REPO_NAME="gr5360-co-project"

cd "$PROJECT_DIR"

if [[ ! -x "$GH_BIN" ]]; then
  echo "GitHub CLI not found at $GH_BIN"
  exit 1
fi

if ! "$GH_BIN" auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated."
  echo "Run:"
  echo "  $GH_BIN auth login --web --git-protocol https --hostname github.com"
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
else
  "$GH_BIN" repo create "$REPO_NAME" --private --source=. --remote=origin --push
fi

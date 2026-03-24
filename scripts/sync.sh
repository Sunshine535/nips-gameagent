#!/usr/bin/env bash
# Three-way sync: Local <-> Server <-> GitHub
#
# Usage:
#   bash scripts/sync.sh push    # local -> server
#   bash scripts/sync.sh pull    # server -> local
#   bash scripts/sync.sh status  # show diff between local and server
#   bash scripts/sync.sh github  # push to GitHub (from local)

set -euo pipefail

SERVER_HOST="wujn@root@ssh-362.default@222.223.106.147"
SERVER_PORT=30022
SERVER_DIR="/gfs/space/private/wujn/Research/nips-gameagent"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_CMD="ssh -p $SERVER_PORT -o StrictHostKeyChecking=no"
RSYNC_OPTS="-avz --progress --exclude='.git/' --exclude='__pycache__/' --exclude='.venv/' --exclude='*.pyc' --exclude='wandb/' --exclude='.specstory/'"

SYNC_CODE_OPTS="$RSYNC_OPTS --exclude='results/' --exclude='data/' --exclude='logs/' --exclude='checkpoints/' --exclude='outputs/' --exclude='*.tar.gz'"

SYNC_RESULTS_OPTS="$RSYNC_OPTS --include='results/***' --include='data/***' --include='logs/***' --exclude='*'"

action="${1:-status}"

case "$action" in
    push)
        echo "=== Pushing code: local -> server ==="
        rsync $SYNC_CODE_OPTS -e "$SSH_CMD" \
            "$LOCAL_DIR/" "$SERVER_HOST:$SERVER_DIR/"
        echo "Done. Code synced to server."
        ;;

    push-all)
        echo "=== Pushing everything: local -> server ==="
        rsync $RSYNC_OPTS -e "$SSH_CMD" \
            "$LOCAL_DIR/" "$SERVER_HOST:$SERVER_DIR/"
        echo "Done. Full sync to server."
        ;;

    pull)
        echo "=== Pulling code: server -> local ==="
        rsync $SYNC_CODE_OPTS -e "$SSH_CMD" \
            "$SERVER_HOST:$SERVER_DIR/" "$LOCAL_DIR/"
        echo "Done. Code synced from server."
        ;;

    pull-results)
        echo "=== Pulling results: server -> local ==="
        rsync $RSYNC_OPTS -e "$SSH_CMD" \
            --include='results/***' --include='logs/***' \
            --include='data/*.json' --exclude='data/*.jsonl' --exclude='*' \
            "$SERVER_HOST:$SERVER_DIR/" "$LOCAL_DIR/"
        echo "Done. Results synced from server."
        ;;

    pull-all)
        echo "=== Pulling everything: server -> local ==="
        rsync $RSYNC_OPTS -e "$SSH_CMD" \
            "$SERVER_HOST:$SERVER_DIR/" "$LOCAL_DIR/"
        echo "Done. Full sync from server."
        ;;

    status)
        echo "=== Comparing local vs server (code only) ==="
        rsync $SYNC_CODE_OPTS --dry-run -e "$SSH_CMD" \
            "$LOCAL_DIR/" "$SERVER_HOST:$SERVER_DIR/" 2>&1 | head -50
        echo ""
        echo "=== Server disk usage ==="
        $SSH_CMD "$SERVER_HOST" "du -sh $SERVER_DIR/*/ 2>/dev/null"
        ;;

    github)
        echo "=== Pushing to GitHub ==="
        cd "$LOCAL_DIR"
        git add -A
        echo "Changes to commit:"
        git status --short
        read -p "Commit message: " msg
        git commit -m "$msg"
        git push origin main
        echo "Done. Pushed to GitHub."
        ;;

    *)
        echo "Usage: $0 {push|push-all|pull|pull-results|pull-all|status|github}"
        exit 1
        ;;
esac

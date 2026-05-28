#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — Start the Emotional AI Agent
#
# What this does:
#   1. Verifies Ollama is installed
#   2. Starts `ollama serve` if it isn't already running
#   3. Waits until the Ollama API is reachable
#   4. Pulls the default model if it isn't already downloaded
#   5. Launches the Streamlit app
#
# The app itself pre-loads whichever model the user profile selects into RAM
# on startup, and unloads the previous model when the user switches models.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
DEFAULT_MODEL="${LLM_MODEL_NAME:-phi3:mini}"
WAIT_SECONDS=30

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

info()    { echo -e "${GREEN}[run]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[run]${RESET} $*"; }
error()   { echo -e "${RED}[run]${RESET} $*" >&2; }

# ── 1. Check Ollama is installed ──────────────────────────────────────────────
if ! command -v ollama &> /dev/null; then
    error "Ollama is not installed."
    error "Install it from https://ollama.com/download, then re-run this script."
    exit 1
fi

# ── 2. Start ollama serve if not already running ──────────────────────────────
is_ollama_up() {
    curl -sf --connect-timeout 2 "$OLLAMA_URL/api/tags" > /dev/null 2>&1
}

if is_ollama_up; then
    info "Ollama is already running at $OLLAMA_URL"
else
    info "Starting Ollama server..."
    # Redirect ollama's own logs so they don't clutter the terminal.
    ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!

    info "Waiting for Ollama to be ready (up to ${WAIT_SECONDS}s)..."
    for i in $(seq 1 $WAIT_SECONDS); do
        sleep 1
        if is_ollama_up; then
            info "Ollama ready after ${i}s  (pid $OLLAMA_PID)"
            break
        fi
        if [ "$i" -eq "$WAIT_SECONDS" ]; then
            error "Ollama did not start within ${WAIT_SECONDS}s."
            error "Check /tmp/ollama.log for details."
            kill "$OLLAMA_PID" 2>/dev/null || true
            exit 1
        fi
    done
fi

# ── 3. Pull the default model if missing ──────────────────────────────────────
model_is_pulled() {
    local model="$1"
    local base
    base=$(echo "$model" | cut -d: -f1)
    curl -sf "$OLLAMA_URL/api/tags" \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
names = [m['name'] for m in data.get('models', [])]
base  = '$base'
model = '$model'
found = any(n == model or n.split(':')[0] == base for n in names)
sys.exit(0 if found else 1)
" 2>/dev/null
}

if ! model_is_pulled "$DEFAULT_MODEL"; then
    info "Model '$DEFAULT_MODEL' not found locally — pulling now..."
    info "(This only happens once; subsequent starts are instant.)"
    ollama pull "$DEFAULT_MODEL"
    info "Pull complete."
else
    info "Model '$DEFAULT_MODEL' is already available."
fi

# ── 4. Launch Streamlit ───────────────────────────────────────────────────────
info "Launching Emotional AI Agent..."
echo ""

cd "$(dirname "$0")"
exec streamlit run app/main.py \
    --server.headless false \
    --browser.gatherUsageStats false \
    "$@"

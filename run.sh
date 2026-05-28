#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — Start the Emotional AI Agent
#
# What this does:
#   1. Verifies Ollama is installed
#   2. Starts `ollama serve` if it isn't already running
#   3. Waits until the Ollama API is reachable
#   4. Offloads every model currently in Ollama RAM (clean slate)
#   5. Reads the user's saved RAM profile from the DB to pick the right model
#   6. Pulls that model if it isn't already downloaded (first run only)
#   7. Launches the Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
FALLBACK_MODEL="${LLM_MODEL_NAME:-phi3:mini}"
WAIT_SECONDS=30

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

info()  { echo -e "${GREEN}[run]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[run]${RESET} $*"; }
error() { echo -e "${RED}[run]${RESET} $*" >&2; }

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

# ── 3. Offload every model currently in Ollama RAM ───────────────────────────
info "Clearing Ollama RAM..."
python3 - <<'PYEOF'
import sys
try:
    import requests
    ps = requests.get("http://localhost:11434/api/ps", timeout=5).json()
    models = ps.get("models", [])
    if not models:
        print("\033[0;32m[run]\033[0m Ollama RAM already clear")
    else:
        for m in models:
            name = m["name"]
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=15,
            )
            print(f"\033[0;32m[run]\033[0m Unloaded: {name}")
except Exception as e:
    print(f"\033[1;33m[run]\033[0m Could not clear models: {e}", file=sys.stderr)
PYEOF

# ── 4. Resolve the model from the saved user profile ─────────────────────────
#
# ram_gb is stored as a plain INTEGER in the DB — no decryption needed.
# Falls back to FALLBACK_MODEL if no profile exists yet.
#
RESOLVED_MODEL=$(python3 - <<PYEOF
import sqlite3, sys
from pathlib import Path

RAM_TO_MODEL = {
    4:  "gemma2:2b-instruct-q4_K_M",
    8:  "gemma2:2b-instruct-q4_K_M",
    16: "mistral",
    32: "qwen2.5:14b",
}

db = Path("$SCRIPT_DIR/data/conversations.db")
if not db.exists():
    sys.exit(0)   # no profile yet; outer script falls back to FALLBACK_MODEL

try:
    conn = sqlite3.connect(str(db), check_same_thread=False)
    row  = conn.execute("SELECT ram_gb FROM user_profile WHERE id='profile'").fetchone()
    conn.close()
    if row:
        model = RAM_TO_MODEL.get(int(row[0]))
        if model:
            print(model)
except Exception as e:
    print(f"warn: {e}", file=sys.stderr)
PYEOF
)

if [ -z "$RESOLVED_MODEL" ]; then
    MODEL="$FALLBACK_MODEL"
    warn "No saved profile found — using fallback model: $MODEL"
else
    MODEL="$RESOLVED_MODEL"
    info "Profile RAM setting → using model: $MODEL"
fi

# ── 5. Pull the model if not already downloaded ───────────────────────────────
model_is_pulled() {
    local target="$1"
    local base
    base=$(echo "$target" | cut -d: -f1)
    curl -sf "$OLLAMA_URL/api/tags" \
        | python3 -c "
import sys, json
data  = json.load(sys.stdin)
names = [m['name'] for m in data.get('models', [])]
base  = '$base'
model = '$target'
found = any(n == model or n.split(':')[0] == base for n in names)
sys.exit(0 if found else 1)
" 2>/dev/null
}

if ! model_is_pulled "$MODEL"; then
    info "Model '$MODEL' not found locally — pulling now..."
    info "(This only happens once; subsequent starts are instant.)"
    ollama pull "$MODEL"
    info "Pull complete."
else
    info "Model '$MODEL' is already available."
fi

# ── 6. Launch Streamlit ───────────────────────────────────────────────────────
info "Launching Emotional AI Agent..."
echo ""

cd "$SCRIPT_DIR"
exec streamlit run app/main.py \
    --server.headless false \
    --browser.gatherUsageStats false \
    "$@"

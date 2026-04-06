#!/usr/bin/env bash
# Initialize vendor dependencies and verify the build environment.
# Run from the repo root: bash scripts/setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHISPER_DIR="$REPO_ROOT/vendor/whisper.cpp"

echo "Repo root: $REPO_ROOT"

# --- whisper.cpp submodule ---
if [ ! -f "$WHISPER_DIR/CMakeLists.txt" ]; then
    echo "Initializing vendor/whisper.cpp submodule..."
    cd "$REPO_ROOT"
    git submodule add https://github.com/ggerganov/whisper.cpp vendor/whisper.cpp || true
    git submodule update --init --recursive
else
    echo "vendor/whisper.cpp already present."
fi

# --- verify cmake + compiler ---
for cmd in cmake g++ make; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install build-essential and cmake."
        exit 1
    fi
done
echo "Build tools OK: cmake $(cmake --version | head -1), $(g++ --version | head -1)"

echo ""
echo "Setup complete. Run scripts/build.sh to compile the pipeline."

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

# --- verify Python (optional, for tooling) ---
if command -v python3 &>/dev/null; then
    PY_VERSION="$(python3 --version 2>&1)"
    echo "Python OK : $PY_VERSION"
else
    echo "WARNING: python3 not found. Python tooling (tools/python/) will not work."
    echo "  Install with: sudo apt install python3"
fi

# --- verify wget (for dataset downloads) ---
if ! command -v wget &>/dev/null; then
    echo "WARNING: wget not found. Dataset downloads will fail."
    echo "  Install with: sudo apt install wget"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. bash scripts/build.sh                  # build the pipeline"
echo "  2. bash scripts/run_tests.sh               # run unit tests (needs -DBUILD_TESTS=ON)"
echo "  3. bash scripts/datasets/download_stage1.sh --lite   # download eval datasets"

